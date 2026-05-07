# Hippo Inner SGD — fwd / inner-grad / outer-bwd 公式推导

实现:`xinhe/model/inner_sgd_triton.py`(`HippoInnerSGD(autograd.Function)`,
forward / backward 两个 `@triton.jit` kernel)。

## 符号

每个 `(b, h, n)` 样本独立。下文所有公式都在单样本视角写;实现时按 BHN 维 batch。

| 量 | shape | 含义 |
|---|---|---|
| `K` | (C, D) | inner SGD 的 keys |
| `V` | (C, D) | inner SGD 的 values(target) |
| `lr` | (C,) | 每 token adaptive lr,即 `loss_weights` |
| `γ` | (D,) | LayerNorm γ(`memory_models.LayerNorm.gamma`) |
| `W₁` | (D, DH) | MemoryMLP 第一层 |
| `W₂` | (DH, D) | MemoryMLP 第二层 |
| `eps` | scalar | LN eps,默认 `1e-5` |

`C = chunk_size = 128`,`D = head_dim = 64`,`DH = D · expansion = 128`。

## 1. Forward

`store_memories` 内 `forward_and_loss(params, K, lr, V)`(原 `neural_memory.py:400`)
的展开,对应 `ResidualNorm(MemoryMLP(depth=2))`:

```
h_pre[c, k] = Σ_d K[c, d] · W₁[d, k]                        # (C, DH)
h[c, k]     = gelu(h_pre[c, k])                              # F.gelu(approximate='none')
raw[c, e]   = Σ_k h[c, k] · W₂[k, e]                         # (C, D)
μ[c]        = (1/D) · Σ_e raw[c, e]                          # (C,)
var[c]      = (1/D) · Σ_e (raw[c, e] - μ[c])²                # (C,)
σ[c]        = √(var[c] + eps)                                # (C,)
ln[c, e]    = (raw[c, e] - μ[c]) / σ[c]                      # (C, D)
pred[c, e]  = ln[c, e] · (γ[e] + 1) + K[c, e]                # (C, D)  ResidualNorm 的残差是 input K
r[c, e]     = pred[c, e] - V[c, e]                            # (C, D)
L_c[c]      = (1/D) · Σ_e r[c, e]²                            # (C,)  per-token unweighted MSE
L           = Σ_c lr[c] · L_c[c]                              # 标量,vmap+grad 真正取梯度的对象
```

GeLU exact(对应 `F.gelu(approximate='none')`):

```
Φ(x)  = 0.5 · (1 + erf(x / √2))
φ(x)  = exp(-x²/2) / √(2π)
G(x)  = x · Φ(x)
G'(x) = Φ(x) + x · φ(x)
G''(x) = (2 - x²) · φ(x)
```

## 2. Inner gradient(`HippoInnerSGD.forward` 输出)

vmap+grad 输出 `∇_params L`,即 `∇γ / ∇W₁ / ∇W₂`,store_memories 取其 `-` 当 surprise:

```
err[c, e]    = (2/D) · lr[c] · r[c, e]                            # (C, D)
∇γ[e]        = Σ_c err[c, e] · ln[c, e]                            # (D,)
g_ln[c, e]   = err[c, e] · (γ[e] + 1)                              # (C, D)
m_g[c]       = (1/D) · Σ_e g_ln[c, e]                              # (C,)
m_gln[c]     = (1/D) · Σ_e g_ln[c, e] · ln[c, e]                   # (C,)
g_raw[c, e]  = (g_ln[c, e] - m_g[c] - ln[c, e] · m_gln[c]) / σ[c]   # (C, D)
∇W₂[k, e]    = Σ_c h[c, k] · g_raw[c, e]                            # (DH, D)
g_h[c, k]    = Σ_e g_raw[c, e] · W₂[k, e]                          # (C, DH)
g_h_pre[c,k] = g_h[c, k] · G'(h_pre[c, k])                         # (C, DH)
∇W₁[d, k]    = Σ_c K[c, d] · g_h_pre[c, k]                          # (D, DH)
```

`HippoInnerSGD.forward` 返回 `(∇γ, ∇W₁, ∇W₂, L_c)`。
saved_for_backward:`K, V, lr, γ, W₁, W₂, h_pre, ln, σ`(其余在 backward 内重算)。

## 3. Outer backward — 二阶 VJP

cotangents `d∇γ (D,) / d∇W₁ (D, DH) / d∇W₂ (DH, D) / dL_c (C,)`,
要返还 `dK (C, D) / dV (C, D) / dlr (C,) / dγ (D,) / dW₁ (D, DH) / dW₂ (DH, D)`。

### 3.1 重算可推量(无需保存)

```
h        = gelu(h_pre)                                      # (C, DH)
pred     = ln · (γ + 1) + K                                  # (C, D)
r        = pred - V                                          # (C, D)
err      = (2/D) · lr[:, None] · r                           # (C, D)
g_ln     = err · (γ + 1)                                     # (C, D)
m_g      = (1/D) · Σ_e g_ln                                  # (C, 1)
m_gln    = (1/D) · Σ_e g_ln · ln                             # (C, 1)
g_raw    = (g_ln - m_g - ln · m_gln) / σ                     # (C, D)
g_h      = g_raw @ W₂ᵀ                                       # (C, DH)
g_h_pre  = g_h · G'(h_pre)                                   # (C, DH)
G_p      = G'(h_pre)                                         # (C, DH)
G_pp     = G''(h_pre)                                        # (C, DH)
```

### 3.2 反向链(按 forward 倒序累加)

```
# (R1) ∇γ → 反向:∇γ[e] = Σ_c err[c,e] · ln[c,e]
adj_ln    = d∇γ[None, :] · err                              # (C, D)
adj_err   = d∇γ[None, :] · ln                               # (C, D)

# (R2) ∇W₁ → 反向:∇W₁[d,k] = Σ_c K[c,d] · g_h_pre[c,k]
adj_K_part1   = g_h_pre @ d∇W₁ᵀ                             # (C, D)
adj_g_h_pre   = K @ d∇W₁                                    # (C, DH)

# (R3) ∇W₂ → 反向:∇W₂[k,e] = Σ_c h[c,k] · g_raw[c,e]
adj_h_part1   = g_raw @ d∇W₂ᵀ                               # (C, DH)
adj_g_raw_p1  = h @ d∇W₂                                    # (C, D)

# (R4) L_c → 反向:L_c[c] = (1/D) Σ_e r[c,e]²
adj_r_part_L  = dL_c[:, None] · (2/D) · r                   # (C, D)

# (R5) Step 20: g_h_pre = g_h · G'(h_pre)
adj_g_h       = adj_g_h_pre · G_p                            # (C, DH)
adj_h_pre     = adj_g_h_pre · g_h · G_pp                     # (C, DH)  -- 后面再加 R19 的贡献

# (R6) Step 19: g_h = g_raw @ W₂ᵀ
adj_g_raw     = adj_g_raw_p1 + adj_g_h @ W₂                  # (C, D)
adj_W₂_part1  = adj_g_hᵀ @ g_raw                             # (DH, D)

# (R7) Step 17: g_raw = (g_ln - m_g - ln · m_gln) / σ
adj_g_ln_R7   = adj_g_raw / σ                                # (C, D)
adj_m_g       = -adj_g_raw.sum(-1, keepdim) / σ              # (C, 1)
adj_ln       += -adj_g_raw · m_gln / σ                       # accumulate
adj_m_gln     = -(adj_g_raw · ln).sum(-1, keepdim) / σ       # (C, 1)
adj_σ_part1   = -(adj_g_raw · g_raw).sum(-1, keepdim) / σ    # (C, 1)

# (R8) Step 16: m_gln = (1/D) Σ_e g_ln · ln
adj_g_ln_R8   = adj_m_gln · ln / D                           # 广播 (C,1)·(C,D)
adj_ln       += adj_m_gln · g_ln / D                          # accumulate

# (R9) Step 15: m_g = (1/D) Σ_e g_ln
adj_g_ln_R9   = adj_m_g / D                                  # (C, 1) 广播到 (C, D)

adj_g_ln      = adj_g_ln_R7 + adj_g_ln_R8 + adj_g_ln_R9      # (C, D)

# (R10) Step 14: g_ln = err · (γ + 1)
adj_err      += adj_g_ln · (γ + 1)                            # accumulate
adj_γ_R10     = (adj_g_ln · err).sum(0)                       # (D,)

# (R11) Step 12: err = (2/D) · lr · r
adj_lr        = (2/D) · (adj_err · r).sum(-1)                 # (C,)  → dlr
adj_r_R11     = (2/D) · lr[:, None] · adj_err                  # (C, D)

adj_r         = adj_r_part_L + adj_r_R11                       # (C, D)

# (R12) Step 9: r = pred - V
adj_pred      = adj_r                                          # (C, D)
adj_V         = -adj_r                                         # (C, D)  → dV

# (R13) Step 8: pred = ln · (γ + 1) + K
adj_ln       += adj_pred · (γ + 1)                              # accumulate
adj_γ_R13     = (adj_pred · ln).sum(0)                          # (D,)
adj_K_R13     = adj_pred                                         # (C, D)

adj_γ         = adj_γ_R10 + adj_γ_R13                            # (D,)  → dγ

# (R14) Step 7: ln = (raw - μ) / σ
adj_raw       = adj_ln / σ                                       # (C, D)
adj_μ         = -(adj_ln.sum(-1, keepdim)) / σ                   # (C, 1)
adj_σ_part2   = -(adj_ln · ln).sum(-1, keepdim) / σ              # (C, 1)

adj_σ         = adj_σ_part1 + adj_σ_part2                         # (C, 1)

# (R15) Step 6: σ = √(var + eps)
adj_var       = adj_σ / (2 · σ)                                  # (C, 1)

# (R16) Step 5: var = (1/D) Σ_e (raw - μ)²
#         ∂var/∂μ = -(2/D) · Σ_e (raw - μ) = -(2/D) · σ · Σ_e ln = 0   ←  Σ_e ln = 0
adj_raw      += adj_var · (2/D) · σ · ln                          # 广播

# (R17) Step 4: μ = (1/D) Σ_e raw
adj_raw      += adj_μ / D                                         # 广播

# (R18) Step 3: raw = h @ W₂
adj_h_R18     = adj_raw @ W₂ᵀ                                     # (C, DH)
adj_W₂_R18    = hᵀ @ adj_raw                                      # (DH, D)

adj_h         = adj_h_part1 + adj_h_R18                            # (C, DH)
adj_W₂        = adj_W₂_part1 + adj_W₂_R18                          # (DH, D)  → dW₂

# (R19) Step 2: h = gelu(h_pre)
adj_h_pre    += adj_h · G_p                                        # accumulate

# (R20) Step 1: h_pre = K @ W₁
adj_K_R20     = adj_h_pre @ W₁ᵀ                                   # (C, D)
adj_W₁        = Kᵀ @ adj_h_pre                                    # (D, DH)  → dW₁

# 汇总
dK  = adj_K_part1 + adj_K_R13 + adj_K_R20                          # (C, D)
dV  = adj_V                                                         # (C, D)
dlr = adj_lr                                                        # (C,)
dγ  = adj_γ                                                         # (D,)
dW₁ = adj_W₁                                                        # (D, DH)
dW₂ = adj_W₂                                                        # (DH, D)
```

### 3.3 关键 invariant

- **`Σ_e ln[c, e] = 0`**(因 ln 是去均值后归一化):用于 R16 简化掉 `adj_μ_var = 0`
- **数值上 `m_g[c] = ε`**(因 `Σ_e g_ln = (γ+1)·Σ_e err`,`Σ_e err = (2/D)·lr·Σ_e r`,
  没有约束 `Σ_e r = 0`,所以 m_g 不是必然 0,**不能省略**)
- **per-(b,h,n) 独立**:整套公式 batch_dim BHN 上完全独立(每样本各自的 W/γ/lr),
  Triton 的 `program_id(0)` 直接索引样本

## 4. 数值稳定性

- 所有 reduction(sum/mean over D)在 Triton 内部用 `tl.float32` 累加,
  即使 caller dtype 是 bf16
- `σ = sqrt(var + eps)` 用 `tl.rsqrt(var + eps)` 等价但数值更稳;同样
  outer bwd 里 `adj_σ / (2σ)` 用 `adj_var = adj_σ · 0.5 · rsqrt_σ²` 等价式
- `gelu / G_p / G_pp` 都用 fp32 求值,结果回 caller dtype

## 5. 与 vmap+grad 的等价性来源

`per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims=(0,0,0,0))`:
- `grad(L, params)` 即上文 `(∇γ, ∇W₁, ∇W₂)` 的精确数学结果
- `vmap` 沿 BHN 0 维 broadcast,等价于"每样本独立 grad"

本算子按 BHN program_id 一比一独立计算,不跨 sample 共享、不平均。
**等价性证明**:每条公式都是从 forward graph 解析手推,与 PyTorch autograd
对同一 forward 的反向传播完全一致(浮点顺序不同会有 ε 级数值漂移,见 tests 容差)。
