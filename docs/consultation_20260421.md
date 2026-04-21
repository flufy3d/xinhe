# Xinhe (心核) Project — Consultation Document

**Date**: 2026-04-21
**Purpose**: seek external AI consultation on architectural direction after 4 failed experiments.

---

## 1. Project Overview

**Goal**: build a transformer that develops emergent "memory" analogous to human cognition:
- `state` = short-term working memory (hippocampus-like), operating during conversation
- `sleep` = long-term memory consolidation (neocortex-like), future milestone

Hardware: Qwen3.5-0.8B backbone, frozen, augmented with:
- LoRA adapters (rank 8, q_proj/v_proj/in_proj_qkv/out_proj)
- External `StateInterface` plugin (~18M params) that acts as persistent slot memory

We want this model to remember facts across multi-turn conversations without writing to a context window — instead, via compressed state that carries between turns.

## 2. Architecture (current best = v5b)

### StateInterface (v5b)
- **n_state = 32** slots, each of `state_dim = 1024`
- `state_emb: (32, 1024)` learnable initial slot table (randn × 0.01)
- **Read side** (per hook layer of backbone):
  - `read_k_projs[l]`, `read_v_projs[l]`: `Linear(state_dim → hidden)` per layer (6 hook layers)
  - `read_scale`: learnable scalar, sigmoid-wrapped, controls how much state influences content
  - At each layer: `hidden += softmax(hidden @ K.T / sqrt(d)) @ V` with `K, V = read_{k,v}_projs[l](state) * scale`
- **Write side** (once per turn, after backbone):
  - `write_q: Linear(state_dim → hidden)`
  - Attention: `attn = softmax(write_q(state_old) @ content.T / sqrt(d))`, shape `(B, n_state, T)`
  - `extracted = attn @ content`
  - `write_out: Linear(hidden → state_dim)`
  - `state_new = write_out(extracted)`
  - `gate_proj: Linear(2*state_dim → state_dim)`, `gate = sigmoid(gate_proj(cat(state_old, state_new)))`
  - `state_next = gate * state_old + (1 - gate) * state_new`
- **Contrastive value head** (v5b key addition):
  - `value_head: Linear(state_dim → hidden)`
  - InfoNCE loss on VALUE tokens: winner slot from write attention should have `value_head(state_next[winner])` close to mean embedding of the value tokens, pushed away from other slots
  - This is a **loss**, not an architectural path — it cannot be "turned off" by a learned gate

### Forward pass flow
```
content_emb = backbone.embed(input_ids)
state_kv = generate_read_kv(state)                       # precompute K/V per layer
content_out = backbone.forward_blocks(
    content_emb,
    layer_hook=lambda h, layer_i: read_layer(h, state_kv[layer_i])
)
state_next = write_from_content(state, content_out)
logits = lm_head(content_out)
loss = weighted_ce(logits, labels, weights) + contrastive_weight * contrastive_loss
```

## 3. Training Setup

### Data
Generated episodic conversations. Each episode:
- Pre-filler: optional small talk (no loss)
- Tell turns: "My name is Alice" / "I work as a doctor" — state must store
- Filler turns: chit-chat between tell and recall
- Recall turns: "What is my job?" → assistant responds "doctor"

Key parameters controlling task difficulty:
- `num_facts`: how many facts per episode (1, 2, 3, 5)
- `entity_ratio`: probability each fact has an entity (Alice/Bob/me/you) vs generic fact
- `same_category`: probability all entities in episode share category (e.g. all are names, forcing entity disambiguation)
- `recall_ratio` / `overwrite_ratio` / `ai_recall_ratio`: recall variations

### Eval metrics
Per-token accuracy split into three classes:
- **VALUE**: recall turn tokens matching the actual answer substring (the real state-dependent correctness metric)
- **FRAME**: recall turn tokens not in the answer (punctuation, template)
- **TELL**: all tokens in tell turns (essentially user parroting — easy)

Plan's success criterion: `VALUE ≥ 99%` at `same_category = 0.5` eval set.

### Curriculum
12 stages, roughly:
- Phase A (stages 0-5, short turn 2-6): single fact → 2 entities same/diff cat → 3-entity multi-fact
- Phase B (stages 6-8, mid turn 7): recall + harder entity
- Phase C (stages 9-11, long turn 16): overwrite, full mix

## 4. Best Current Results (v5b)

| Task | VALUE accuracy | Notes |
|------|----------------|-------|
| Stage 0 (1 name, 2 turns) | 98.44% | solid baseline |
| Stage 1 (1 fact, distance=1) | 97.24% | |
| Stage 2 (2 diff-cat entities) | 99.12% | entity routing easy case |
| **Stage 3 (2 same-cat entities) fine-tune peak** | **91.43%** | 2-entity hard routing |
| Stage 3 cat=0 sweep (200 ep) | 93.64% | |
| Stage 3 **cat=0.5** sweep (200 ep) | **87.62%** | plan primary target condition |
| Stage 3 cat=1.0 sweep (200 ep) | 82.55% | stretch |
| **Stage 5 (3 entities same_cat=0.5) fine-tune** | **~84% plateau** | **the blocker** |

Slot utilization: all 32 slots active (L2 norm > threshold 100% of time). Mean slot norm ~97 (v5a no-contrastive was ~37 — contrastive strengthens state).

Error breakdown at cat=1.0: **72.5% of VALUE errors are "routing errors"** (predicted token belongs to another entity's value in the same episode). Contrastive fixed value semantics; remaining errors concentrate in slot routing precision.

## 5. Diagnosis — The Core Blocker

The fundamental difficulty: **same-category entity disambiguation under generalization**.

Concretely: model sees 10K training episodes with random names like "Alice is a doctor, Bob is a lawyer". Must learn a **function** `(entity_name_in_content, entity_identity_index) → slot_id` that generalizes to new names in val.

When both entities are same category (both names), the model cannot rely on category features to differentiate. It must route by entity identity, which is weakly represented in content attention.

Train/val gap is huge: train often 100% accuracy, val stuck at 80-90%. Model memorizes training names, doesn't generalize routing function.

Contrastive loss helps significantly (v5a → v5b: cat=0.5 went 82.43% → 87.62%, +5.19pp). But hits a ceiling.

## 6. Failed Experiments (2026-04-21)

All 4 architectural attempts to break past v5b's ~84% ceiling on 3-entity tasks **failed**.

### v5c: Single-pass Slot Attention
**Change**: `write_from_content` softmax dim -1 → -2 (slots compete for tokens instead of tokens compete for slots).
**Result**: warm-start from v5b ckpt, stage 3 collapsed from 80% to 55-70%.
**Root cause**: the model's `write_q` was trained for per-slot softmax semantics; slot-attention softmax produces different attention shapes. Warm-start broke.

### v5d: Write Iterations = 2
**Change**: iterate write_from_content 2x per segment, updating state with simple gated linear combination.
**Result**: catastrophic, stage 3 step 1000 = 44%.
**Root cause**: iteration 2's state_old is "noisy intermediate from iter 1" — a distribution write_q was never trained on. Garbage Q, garbage update, overwrites iter 1 result.

### v5e: Multi-Head Write Attention (4 heads)
**Change**: add `write_k`, `write_v` projections and split Q/K/V into 4 heads.
**Result**: fresh train, stage 2 collapsed at end (step 1500: 95%, step 2000: 8.41% — 12x loss explosion in 500 steps). Even with LR min clamp at 5%, Adam optimizer diverged at low LR. Stage 3 step 1000 = 55%.
**Root cause 1**: Adam `eps=1e-8` too small: at low LR + long stable training, v_hat decays to ~1e-10; any gradient spike produces `update = grad/sqrt(v_hat) ≈ 1500`, catastrophic.
**Root cause 2**: 4 heads on 2-entity task dilutes signal. More capacity ≠ better on simple tasks.

### v5f: Full Slot Attention + GRU + LayerNorm + MLP (Locatello 2020 faithful implementation)
**Change**: replace write path with canonical Slot Attention:
- to_q, to_k, to_v projections
- 3 iterations of slot attention
- `softmax(dim=-2)` with per-token normalization
- `nn.GRUCell` for slot updates (not linear gate — reset gate allows "ignore noisy prev")
- LayerNorm before Q, MLP residual after GRU
- Orthogonal init for state_emb

**Result** (fresh train): stage 2 end collapsed to VALUE=0%, FRAME=0%, TELL=54% (different collapse pattern from v5e). Re-ran with `max_steps` cut to 1500 to avoid divergence window: stage 2 end 80.53%. Stage 3 **plateaued at 55%** through step 3000. v5b matches this in ~300 steps.

**Root causes**:
1. Same Adam instability as v5e (LR end).
2. More fundamentally: Slot Attention + GRU + 3 iterations trains **much slower** than single-head cross-attention. For simple 2-entity tasks, complexity overhead exceeds benefit. Only theoretically useful at 3+ entities — but we never get there because it can't even match v5b on 2.

### Common patterns across failures
1. Architecture changes via warm-start from v5b ckpts don't work (write_q learned for v5b mechanism, new semantics need fresh training).
2. Fresh trains with more complex architectures consistently underperform v5b in early stages, creating a "debt" that doesn't recover.
3. **Optimizer instability**: Adam + cosine LR → near-0 + contrastive loss + complex architecture = loss explodes at stage end. Partially mitigated by `max(0.1, cosine)` LR floor, not fully cured.

## 7. Working Hypothesis for Consultation

After 4 failed slot-routing tweaks, I believe the fundamental bottleneck isn't in the write mechanism. It's that **compressed slot storage with learned routing is the wrong abstraction for episodic facts**.

Proposed pivot: **Episodic Memory architecture** (hippocampus-inspired):

### Design sketch
```
state: (32, 1024)                         # compressed semantic (existing, keep)
episodic_buffer: list[(key, value)]       # NEW, verbatim storage, size L ~256

On tell turn:
  (key, value) = encode(content)
  episodic_buffer.append((key, value))    # no routing, no loss of info
  (slot_state also updated normally as existing)

On recall turn:
  query = encode(question)
  sim = query @ buffer_keys.T
  attn = softmax(sim / tau)
  retrieved = attn @ buffer_values
  hidden = hidden + α * cross_attn(query=hidden, kv=slot_state) + β * retrieved
```

### Why it might work where slot routing failed
- Buffer retrieval is **key-similarity**, which is what transformer attention already does well
- No routing precision needed — Alice's (key, value) is literally in buffer, retrievable by similarity of question
- Slot state remains for semantic "what's this conversation generally about" background
- Maps directly onto user's `state` (short-term) vs `sleep` (long-term consolidation) vision

### Open questions we need consultation on

1. **Is episodic memory the right move?** Or is there a better way to rescue the slot-routing approach? Is there a specific architectural fix that addresses "same-category entity routing" that we haven't tried?

2. **If episodic memory**: how to encode (key, value)?
   - Option A: use backbone token embeddings directly (mean-pool over tokens)
   - Option B: learned encoder head
   - Option C: hybrid — key from learned head (for semantic matching), value verbatim tokens
   - Which is proven to work in similar settings?

3. **Buffer management**:
   - Fixed size L, FIFO? LRU? learned eviction?
   - Consolidation to slot state — what's the learned signal for when / what to consolidate?
   - Is there prior art for this in transformer literature (Memory Networks, Retentive Networks, Differentiable Neural Computers, Transformer-XL, Hopfield Networks, etc.) that we should study?

4. **Training dynamics**:
   - Adam + cosine LR → 0 combined with new architecture consistently caused late-stage divergence. What's the robust optimizer/scheduler for transformer + memory augmentation?
   - Should we use something other than AdamW (Adafactor, Lion)?

5. **Curriculum for generalization**:
   - Current curriculum is stage-based with increasing difficulty. One hypothesis (from user): having stages where entity routing is NOT needed (e.g. pure fact storage) causes the model to "forget" entity routing skill learned earlier. Would a uniform curriculum (every episode mixes all difficulty dimensions) work better for generalization?

## 8. Constraints / Context

- Training hardware: RTX 4090 24GB remote, ~5-6 hours per full curriculum run
- Model size fixed at 0.8B (need 99% here before scaling to 4B)
- Must not hardcode task-specific heuristics (e.g. explicit entity-token detection via rules) — model must learn from raw content
- User's long-term vision: state = online learning that persists per conversation, sleep = periodic consolidation from replay buffer
- Project philosophy: "emergent memory from unified state in small transformers"

## 9. Summary Ask

Given 4 failed architecture experiments on slot routing, we want to know:

1. **Validate / criticize** the hypothesis that slot-based compressed routing is hitting a fundamental ceiling for generalizable entity disambiguation.

2. **Recommend** either:
   (a) a specific architectural fix to slot routing that we haven't tried and that has theoretical grounding for breaking the plateau, or
   (b) a concrete design for episodic memory augmentation that would work at our scale (0.8B model, 32-slot state, target 99% on 3-entity same-category tasks).

3. **Identify prior art** we should study: any papers solving essentially this problem (multi-entity binding in memory-augmented transformers that generalizes across paraphrases)?

4. **Optimizer / training stability** advice for memory-augmented transformers.
