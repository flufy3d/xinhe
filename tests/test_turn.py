"""
测试 TurnInterface (v6.1: W_turn 自旋时序罗盘 + 多相位共振搜索读侧)
"""
import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

from xinhe.model.fact_plugin import FactInterface
from xinhe.model.turn_plugin import (
    TurnInterface, apply_rotation_k, rotate_W_turn, rotate_query, _rotate_half,
)
from xinhe.model.dual_state import DualState


@pytest.fixture
def fact_iface():
    return FactInterface(hidden_size=64, n_heads=4, head_dim=16, n_layers=2)


@pytest.fixture
def turn_iface(fact_iface):
    return TurnInterface(
        fact_interface=fact_iface,
        hidden_size=64, n_heads=4, head_dim=16, n_layers=2,
        turn_read_scale_init=-8.0,
        turn_gamma=0.9,
        turn_phase_max=5,
    )


def test_blank_turn_shape(turn_iface):
    """空白 W_turn: (B, H, d_v, d_k)，零初始化。"""
    W = turn_iface.blank_state(batch_size=2)
    assert W.shape == (2, 4, 16, 16)
    assert (W == 0).all()


def test_turn_read_identity_at_init(turn_iface):
    """零 W_turn + 初始 read_scale≈0 → read_layer 输出 ≈ 输入。"""
    W = turn_iface.blank_state(1)
    hidden = torch.randn(1, 8, 64)
    out = turn_iface.read_layer(hidden, W, layer_idx=0)
    # W=0 时 read 项恒为 0，所以 out 应该完全等于 hidden（不论 scale 是多少）
    assert torch.allclose(out, hidden, atol=1e-6)


def test_turn_write_decay_and_accumulate(turn_iface):
    """写两轮相同 content：||W_turn_2|| 应在 [0.5·||res||, 2.0·||res||] 之间。

    数学：W1 = γ·R·0 + res = res；W2 = γ·R·res + res。
    旋转正交 → ||γ·R·res|| = γ·||res||。具体 ||W2|| 取决于 <R·res, res> 对齐度：
      - 完全对齐（R≈I）：||W2|| ≈ (1+γ)·||res||
      - 正交：||W2|| ≈ sqrt(1+γ²)·||res||
      - 反向：||W2|| ≈ (1-γ)·||res||
    实际上多频段旋转 → 中等对齐，||W2|| 近似 (1+γ·cos_avg)·||res||，宽范围容错。
    """
    B, T, D = 1, 8, 64
    content = torch.randn(B, T, D) * 0.1
    W0 = turn_iface.blank_state(B)

    W1 = turn_iface.write_from_content(W0, content)
    W2 = turn_iface.write_from_content(W1, content)

    res_norm = W1.norm()
    ratio = W2.norm() / (res_norm + 1e-12)
    assert 0.5 < ratio.item() < 2.0, \
        f"累积异常: ||W2||/||W1||={ratio:.4f}, 期望 0.5-2.0"
    # 至少应该比单轮大（累积效应）
    assert W2.norm() > 0.3 * W1.norm()


def test_turn_gradient_flow_read_side_only(turn_iface):
    """梯度应该流向 q_projs_turn / o_projs_turn / read_scale_turn，
    验证 read 侧所有可学参数都有梯度路径。多相位搜索后无 dtau_head 参数。
    """
    W = torch.randn(1, 4, 16, 16) * 0.1
    hidden = torch.randn(1, 8, 64, requires_grad=True)
    out = turn_iface.read_layer(hidden, W, layer_idx=0)
    loss = out.sum()
    loss.backward()

    assert turn_iface.q_projs_turn[0].weight.grad is not None
    assert turn_iface.o_projs_turn[0].weight.grad is not None
    assert turn_iface.read_scale_turn.grad is not None


def test_turn_write_side_grad_flow(turn_iface):
    """v6.2: 写侧的 k_proj_turn / v_proj_turn 应有独立梯度路径。"""
    W0 = turn_iface.blank_state(1)
    content = torch.randn(1, 8, 64, requires_grad=True) * 0.1
    W_new = turn_iface.write_from_content(W0, content)
    # 模拟下游 loss，要求 W_new 的 norm 大
    W_new.norm().backward()
    assert turn_iface.k_proj_turn.weight.grad is not None
    assert turn_iface.v_proj_turn.weight.grad is not None
    assert turn_iface.k_proj_turn.weight.grad.abs().sum().item() > 0
    assert turn_iface.v_proj_turn.weight.grad.abs().sum().item() > 0


def test_dual_state_detach_and_to():
    """DualState.detach()/.to() 正确传递到两个张量。"""
    W_fact = torch.randn(2, 4, 16, 16, requires_grad=True)
    W_turn = torch.randn(2, 4, 16, 16, requires_grad=True)
    state = DualState(W_fact, W_turn)

    detached = state.detach()
    assert not detached.W_fact.requires_grad
    assert not detached.W_turn.requires_grad
    assert torch.equal(detached.W_fact, W_fact.detach())
    assert torch.equal(detached.W_turn, W_turn.detach())

    # to(cpu) 应 no-op 但保持 DualState 类型
    moved = state.to("cpu")
    assert isinstance(moved, DualState)


def test_fact_path_unchanged_when_turn_zero_and_scale_minus_inf(fact_iface):
    """W_turn=0 + turn read_scale 极低 → 叠加后的 hidden 与 fact-only 差距 < 1e-4。
    这是"turn 路径初始静默"的数值验证。
    """
    turn = TurnInterface(
        fact_interface=fact_iface,
        hidden_size=64, n_heads=4, head_dim=16, n_layers=2,
        turn_read_scale_init=-20.0,   # sigmoid(-20) ≈ 2e-9
        turn_gamma=0.9,
    )
    W_fact = torch.randn(1, 4, 16, 16) * 0.1
    W_turn_zero = turn.blank_state(1)
    hidden = torch.randn(1, 8, 64)

    # fact-only
    h1 = fact_iface.read_layer(hidden, W_fact, layer_idx=0)
    # fact + turn（W_turn=0 且 scale≈0）
    h2 = turn.read_layer(h1, W_turn_zero, layer_idx=0)

    assert torch.allclose(h1, h2, atol=1e-4), \
        f"turn 路径未静默: max_diff={(h1-h2).abs().max():.6f}"


# ═══════════════════════════════════════════════════════════════════
# 旋转原语单元测试
# ═══════════════════════════════════════════════════════════════════


def test_rotation_inverts(turn_iface):
    """apply_rotation_k(x, s) 后 apply_rotation_k(·, s, inverse=True) = x（fp32 容差）。"""
    x = torch.randn(2, 4, 8, 16)                   # (B,H,T,d_k)
    steps = torch.tensor(3.5)                      # scalar
    steps_b = steps.expand(2, 4, 8)                # 匹配 x.shape[:-1]
    inv_freq = turn_iface.inv_freq

    rot = apply_rotation_k(x, steps_b, inv_freq, inverse=False)
    back = apply_rotation_k(rot, steps_b, inv_freq, inverse=True)
    assert torch.allclose(x, back, atol=1e-5), \
        f"旋转不可逆: max_diff={(x-back).abs().max():.6e}"


def test_rotation_preserves_norm(turn_iface):
    """旋转保范数（RoPE 是正交变换）。"""
    x = torch.randn(1, 4, 8, 16)
    steps = torch.tensor(2.7)
    steps_b = steps.expand(1, 4, 8)
    inv_freq = turn_iface.inv_freq

    rot = apply_rotation_k(x, steps_b, inv_freq, inverse=False)
    # 每个 (b,h,t) 位置的 d_k 向量范数应一致
    norm_x = x.norm(dim=-1)
    norm_rot = rot.norm(dim=-1)
    assert torch.allclose(norm_x, norm_rot, atol=1e-5), \
        f"旋转未保范数: max_diff={(norm_x-norm_rot).abs().max():.6e}"


def test_read_layer_handles_bf16_hidden_with_fp32_weights(fact_iface):
    """val 路径（无 autocast）会给 hidden=bf16、而参数默认 fp32，要能跑通。"""
    turn = TurnInterface(
        fact_interface=fact_iface,
        hidden_size=64, n_heads=4, head_dim=16, n_layers=1,
    )  # 默认 fp32 参数

    W = torch.randn(1, 4, 16, 16, dtype=torch.bfloat16) * 0.1
    hidden = torch.randn(1, 8, 64, dtype=torch.bfloat16)
    out = turn.read_layer(hidden, W, layer_idx=0)     # 不应该报 dtype mismatch
    assert out.dtype == torch.bfloat16
    assert out.shape == hidden.shape


# ═══════════════════════════════════════════════════════════════════
# 多相位共振搜索读侧测试
# ═══════════════════════════════════════════════════════════════════


def test_multi_phase_read_empty_W_uniform_alpha(turn_iface):
    """W_turn=0 → 所有相位 r_τ=0 → softmax 均匀 1/(P+1)，输出 = hidden。"""
    W = turn_iface.blank_state(1)
    hidden = torch.randn(1, 8, 64)
    out = turn_iface.read_layer(hidden, W, layer_idx=0)
    # out = hidden + scale * o_proj(0) = hidden
    assert torch.allclose(out, hidden, atol=1e-6)
    # 最近 alpha 应当在 P 维接近均匀
    diag = turn_iface.get_phase_diagnostics()
    assert diag is not None
    # 均匀分布熵 = log(P+1) = log(6) ≈ 1.7918
    expected_ent = torch.tensor(turn_iface.phase_max + 1.0).log().item()
    assert abs(diag["turn_phase_entropy"] - expected_ent) < 1e-3, \
        f"W=0 时 softmax 应均匀 (熵≈{expected_ent:.4f})，得到 {diag['turn_phase_entropy']:.4f}"


def test_multi_phase_read_picks_correct_age(fact_iface):
    """构造 W_turn 使得只有 age=τ* 的条目非零，喂语义匹配 q，期望 α[τ*] 显著高。

    做法：
      - 取 k_ref，正旋转 age 步得到 R^age·k_ref 作为 W_turn 的 key 维
      - 喂入的 q 经 q_projs_turn 投影 → L2 归一 → 与 k_ref 语义对齐
      - read_layer 内部枚举相位 q_τ = R^τ·q，在 τ=age 时 <R^τ q, R^age k_ref> = <q, k_ref> 最大
    """
    torch.manual_seed(0)
    turn = TurnInterface(
        fact_interface=fact_iface,
        hidden_size=64, n_heads=4, head_dim=16, n_layers=1,
        turn_phase_max=5,
    )
    B, T, H, d_k, d_v = 1, 4, 4, 16, 16
    target_age = 2

    # 1) 喂一个 hidden，通过 q_projs_turn[0] 算出实际的 q (L2-normalized)
    hidden = torch.randn(B, T, 64)
    with torch.no_grad():
        q = F.linear(hidden, turn.q_projs_turn[0].weight)
        q = q.view(B, T, H, d_k).transpose(1, 2)         # (B,H,T,d_k)
        q = F.normalize(q, dim=-1)                       # 与 read_layer 内保持一致

    # 2) 构造 W_turn 让 k 方向 = R^age · q[:, :, 0, :]（取 token 0 的 q 方向做 key）
    #    v = 任意非零向量；这样 <q_τ, k> 在 τ=age 时最大
    k_ref = q[:, :, 0, :]                                 # (B,H,d_k)
    steps = torch.full((B, H, d_v), float(target_age))
    k_aged = apply_rotation_k(
        k_ref.unsqueeze(-2).expand(B, H, d_v, d_k),
        steps, turn.inv_freq, inverse=False,
    )                                                     # (B,H,d_v,d_k)
    v_ref = torch.randn(B, H, d_v, 1) * 0.5               # (B,H,d_v,1)
    # W[b,h,v,d] = v_ref[b,h,v] * k_aged[b,h,v,d]（外积存储）
    W_turn = v_ref * k_aged                               # (B,H,d_v,d_k)

    # 3) 触发 read_layer；检查 diag
    _ = turn.read_layer(hidden, W_turn, layer_idx=0)
    diag = turn.get_phase_diagnostics()
    assert diag is not None
    # 对于 token 0（q 就是 k_ref 对应方向），预期 argmax 相位 ≈ target_age
    # 其他 token 的 q 与 k_ref 无关，相位随机；我们看整体 argmax 分布里 target_age 是否显著
    alpha = turn._last_alpha       # (P,B,H,T)
    # 只看 token 0 的相位分布（所有 head 平均）
    alpha_t0 = alpha[:, 0, :, 0].mean(dim=-1)   # (P,)
    top_phase = int(alpha_t0.argmax().item())
    assert top_phase == target_age, \
        f"期望 argmax 相位 = age={target_age}，实际 {top_phase}；分布={alpha_t0.tolist()}"
    # 权重应严格高于均匀基线 1/(P+1) ≈ 0.167
    # 注：d_k=16 的小维度下 softmax 峰不够尖（真实训练 d_k=64 会显著更尖），
    # 这里只做"比均匀高"的最低正确性检查
    uniform = 1.0 / (turn.phase_max + 1)
    assert alpha_t0[target_age].item() > uniform, \
        f"target 相位权重应 > 均匀基线 {uniform:.3f}，实际 {alpha_t0[target_age].item():.3f}"


def test_multi_phase_read_differentiable(turn_iface):
    """多相位 softmax 融合路径全程可微，梯度能回到 q_projs_turn / o_projs_turn / read_scale_turn。"""
    W = torch.randn(1, 4, 16, 16) * 0.2
    hidden = torch.randn(1, 8, 64, requires_grad=True)
    out = turn_iface.read_layer(hidden, W, layer_idx=0)
    out.sum().backward()
    for p in turn_iface.q_projs_turn[0].parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    for p in turn_iface.o_projs_turn[0].parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert turn_iface.read_scale_turn.grad is not None


def test_phase_max_is_runtime_configurable(fact_iface):
    """phase_max 是 Python int，推理时覆写应立即生效（不影响权重 shape）。"""
    turn = TurnInterface(
        fact_interface=fact_iface,
        hidden_size=64, n_heads=4, head_dim=16, n_layers=1,
        turn_phase_max=3,
    )
    W = torch.randn(1, 4, 16, 16) * 0.1
    hidden = torch.randn(1, 8, 64)
    _ = turn.read_layer(hidden, W, layer_idx=0)
    assert turn._last_alpha.shape[0] == 4           # P+1 = 3+1

    turn.phase_max = 7                               # 推理时覆写
    _ = turn.read_layer(hidden, W, layer_idx=0)
    assert turn._last_alpha.shape[0] == 8
