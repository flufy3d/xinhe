"""Phase 3 Gated DeltaNet 写 kernel 数学正确性(CPU fp64,无需 GPU)。

覆盖:chunked≡recurrent 等价、带入初始 state、overwrite 删旧关联(delta vs Hebbian)、
关联记忆恢复、150 distract 谱稳定、gated g<1 衰减、平台 flag。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xinhe.model.delta_kernel import (
    torch_delta_chunk, torch_delta_recurrent, _FLA_AVAILABLE, _resolve_backend,
)

F64 = torch.float64


def _rand_kvb(B, H, T, dk, dv, seed=0, beta_scale=0.9):
    torch.manual_seed(seed)
    k = F.normalize(torch.randn(B, H, T, dk, dtype=F64), dim=-1)
    v = torch.randn(B, H, T, dv, dtype=F64)
    beta = torch.rand(B, H, T, dtype=F64) * beta_scale
    return k, v, beta


def test_chunk_equals_recurrent_fp64():
    k, v, beta = _rand_kvb(2, 3, 16, 8, 8)
    W0 = torch.zeros(2, 3, 8, 8, dtype=F64)
    Wc = torch_delta_chunk(W0, k, v, beta)
    Wr = torch_delta_recurrent(W0, k, v, beta)
    assert (Wc - Wr).abs().max().item() < 1e-9


def test_chunk_carries_initial_state():
    k, v, beta = _rand_kvb(1, 2, 8, 4, 4, seed=3)
    W0 = torch.randn(1, 2, 4, 4, dtype=F64) * 0.1
    Wc = torch_delta_chunk(W0, k, v, beta)
    Wr = torch_delta_recurrent(W0, k, v, beta)
    assert (Wc - Wr).abs().max().item() < 1e-9


def test_overwrite_deletes_old_association():
    """写 (k,v1) 再写同 k (k,v2),retrieve k → v2(删旧),非 Hebbian 的 v1+v2。"""
    kv = F.normalize(torch.randn(1, 1, 1, 4, dtype=F64), dim=-1)
    ks = torch.cat([kv, kv], dim=2)
    vs = torch.tensor([[[[1., 0, 0, 0], [0, 1, 0, 0]]]], dtype=F64)
    bts = torch.ones(1, 1, 2, dtype=F64)
    W = torch_delta_chunk(torch.zeros(1, 1, 4, 4, dtype=F64), ks, vs, bts)
    r = torch.einsum("bhvd,bhd->bhv", W, kv[:, :, 0]).flatten()
    assert abs(r[0].item()) < 1e-6 and abs(r[1].item() - 1) < 1e-6   # ≈ v2=[0,1,0,0]


def test_write_then_retrieve_recovers_values():
    """正交 key + β=1 → retrieve k_i 精确恢复 v_i(关联记忆 sanity)。"""
    K, d = 6, 16
    torch.manual_seed(5)
    k = torch.zeros(1, 1, K, d, dtype=F64)
    for i in range(K):
        k[0, 0, i, i] = 1.0                       # 标准基,严格正交
    v = torch.randn(1, 1, K, d, dtype=F64)
    beta = torch.ones(1, 1, K, dtype=F64)
    W = torch_delta_chunk(torch.zeros(1, 1, d, d, dtype=F64), k, v, beta)
    for i in range(K):
        r = torch.einsum("bhvd,bhd->bhv", W, k[:, :, i]).flatten()
        assert F.cosine_similarity(r, v[0, 0, i], dim=0).item() > 0.999


def test_spectral_stable_under_150_distract():
    """β<1 + ‖k‖=1 → 每步收缩,M 谱范数有界(Hebbian 累加会随 √T 增长)。"""
    k, v, beta = _rand_kvb(1, 1, 150, 8, 8, seed=1)
    W = torch_delta_chunk(torch.zeros(1, 1, 8, 8, dtype=F64), k, v, beta)
    sn = torch.linalg.matrix_norm(W[0, 0], ord=2).item()
    assert sn < 50


def test_gated_g_differs_from_plain():
    k, v, beta = _rand_kvb(1, 1, 6, 4, 4, seed=2)
    W0 = torch.zeros(1, 1, 4, 4, dtype=F64)
    g = torch.full((1, 1, 6), 0.5, dtype=F64)
    Wg = torch_delta_recurrent(W0, k, v, beta, g=g)
    Wp = torch_delta_recurrent(W0, k, v, beta)
    assert Wg.shape == W0.shape and (Wg - Wp).abs().max().item() > 1e-6


def test_fla_flag_matches_platform():
    if sys.platform != "linux":
        assert _FLA_AVAILABLE is False
    # auto 在 CPU 上(无 cuda)必回退 torch
    assert _resolve_backend("auto", torch.zeros(1, 1, 2, 2)) == "torch"


def test_explicit_fla_raises_when_unavailable():
    if not _FLA_AVAILABLE:
        with pytest.raises(RuntimeError):
            _resolve_backend("fla", torch.zeros(1, 1, 2, 2))
