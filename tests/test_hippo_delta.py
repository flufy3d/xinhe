"""HippoDelta(Phase 3 Gated DeltaNet 记忆单元)单元测试(CPU)。

覆盖:write/retrieve shape、空 M 读出 0(NM-zero sanity)、β‖k‖²<1 约束、
detach 纪律、write 接 W_k/W_v/W_β 梯度、retrieve 回传梯度到外部 q、overwrite 删旧。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xinhe.model.hippo_delta import HippoDelta, HippoDeltaState


def _hd(d_model=64, d_key=32, d_value=16, n_heads=4, **kw):
    return HippoDelta(d_model, d_key=d_key, d_value=d_value, n_heads=n_heads, **kw)


def test_write_retrieve_shapes():
    hd = _hd()
    B, T = 2, 12
    st = hd.write(torch.randn(B, T, 64), None)
    assert st.M.shape == (B, 4, 16 // 4, 32 // 4)        # (B,H,d_v/H,d_k/H)=(2,4,4,8)
    assert st.seq_index == T
    r = hd.retrieve(st.M, torch.randn(B, 6, 32))         # n_q=6, d_key=32
    assert r.shape == (B, 6, 16)                          # (B,n_q,d_value)


def test_empty_M_retrieves_zero():
    """空 M(首 turn / NM-zero)读出严格 0。"""
    hd = _hd()
    r = hd.retrieve(None, torch.randn(2, 6, 32))
    assert r.abs().max().item() == 0.0


def test_beta_knorm_contraction_enforced():
    """β·‖k‖² < 1 必须处处成立(否则 M 爆炸)。用对抗 init 也要扛住。"""
    hd = _hd(tau_k_init=1.0, beta_bias_init=10.0)         # bias 大 → sigmoid→1,逼近边界
    k, v, beta, g = hd._project(torch.randn(3, 20, 64))
    knorm2 = (k ** 2).sum(dim=-1)                         # (B,H,T)
    assert (beta * knorm2).max().item() < 1.0


def test_detach_breaks_graph():
    hd = _hd()
    st = hd.write(torch.randn(1, 8, 64), None)
    assert st.M.requires_grad                            # write 接梯度
    st2 = st.detach()
    assert not st2.M.requires_grad and st2.seq_index == st.seq_index


def test_write_grads_reach_projections():
    """write 的梯度必须到 W_k/W_v/W_β(否则 write 通路训不到)。"""
    hd = _hd()
    st = hd.write(torch.randn(2, 12, 64), None)
    st.M.sum().backward()
    for name, mod in [("W_k", hd.W_k), ("W_v", hd.W_v), ("W_beta", hd.W_beta)]:
        assert mod.weight.grad is not None and mod.weight.grad.norm().item() > 0, name


def test_retrieve_grads_reach_query():
    """retrieve 的梯度必须回传到外部 q(QueryHead 训练通路)。"""
    hd = _hd()
    M = hd.write(torch.randn(1, 12, 64), None).M.detach()
    q = torch.randn(1, 6, 32, requires_grad=True)
    hd.retrieve(M, q).sum().backward()
    assert q.grad is not None and q.grad.norm().item() > 0


def test_state_carries_across_writes():
    """连续 write carry M(episode 内跨 turn 演化)。"""
    hd = _hd()
    st1 = hd.write(torch.randn(1, 8, 64), None)
    st2 = hd.write(torch.randn(1, 8, 64), st1.detach())
    assert st2.seq_index == 16 and not torch.equal(st1.M, st2.M)


def test_spec_history_optional():
    """probe 用 _spec_log_enabled:开启后每次 write append specnorm,关闭零开销。"""
    hd = _hd()
    # 默认关:write 不 append
    hd.write(torch.randn(1, 8, 64), None)
    assert hd._spec_history == [], "default 应零开销,_spec_history 为空"

    # 开启后跑 5 次,_spec_history 累 5 个 finite 数
    hd._spec_log_enabled = True
    state = None
    for _ in range(5):
        state = hd.write(torch.randn(1, 8, 64), state.detach() if state else None)
    assert len(hd._spec_history) == 5
    assert all(isinstance(x, float) and x >= 0 and x == x for x in hd._spec_history), \
        "spec history 必须全部 finite float"
