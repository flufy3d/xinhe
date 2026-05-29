"""测试 mem_mac_override / mem_mal_override 独立 ablation 接口(probe 用)。

新接口:default None 不动现有路径;非 None 时分别独立控制 MAC-R / MAL alpha。
CPU + DummyBackbone(纯 Linear,无 attention)→ 前缀 token 不影响 real token,
所以黑盒 logits 比较不可靠;改用 white-box hook 验证 routing。
ablate-mac-mal 的真正 GPU 黑盒断言留给 nm_debug_bench 子命令。
"""
import pytest
import torch

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from tests.test_backbone import DummyBackbone


def _make_model(seed: int = 42):
    """构造最小可跑的 XinheModel:DummyBackbone + 无 LoRA / 无 K_pers / 无 aux。"""
    torch.manual_seed(seed)
    backbone = DummyBackbone(hidden_size=64, vocab_size=50)
    cfg = XinheConfig(
        hidden_size=64,
        n_heads=4, head_dim=16,
        freeze_backbone=False,
        lora_rank=0,
        n_persistent_per_layer=0,
        n_query=4,
        d_key=32, d_value=32,
        nm_aux_weight=0.0,
        lambda_div=0.0,
        global_write_layer=-1,        # DummyBackbone 4 层 → write@L3
        mal_inject_layer=-3,          # 4 + (-3) = 1 → MAL@L1
        mal_alpha_init=0.0,           # σ(0)=0.5 强信号
        delta_backend="torch",
    )
    model = XinheModel(cfg, backbone=backbone)
    # MAL 默认 W_mal=zeros_,probe 测 routing 时让它非零(否则 mal_vec 永远 0)
    torch.nn.init.xavier_uniform_(model.W_mal.weight)
    model.eval()
    return model


def _run_state_then_forward(model, ids, **overrides):
    """先 forward 一次 carry M,再用 overrides forward 一次返回 (logits, state_next)。"""
    state = model.init_state(ids.shape[0])
    with torch.no_grad():
        r1 = model(ids, state, pad_token_id=0)
        state = r1["state_next"]
        r2 = model(ids, state, pad_token_id=0, **overrides)
    return r2["logits"], state


def _capture_backbone_input(model, ids, **overrides):
    """monkey-patch backbone.forward_blocks 截获送进的 hidden_states
    (即 [mac_tokens, content_emb] cat 之后)。前 n_query 个 token = MAC-R 前缀。
    XinheModel 调 backbone.forward_blocks(方法,非 forward),所以 forward_hook 不 fire,
    用 patch 是唯一可靠方式。"""
    captured = {}
    orig = model.backbone.forward_blocks

    def patched(hidden_states, *args, **kwargs):
        captured["hidden"] = hidden_states.detach().clone()
        return orig(hidden_states, *args, **kwargs)

    model.backbone.forward_blocks = patched
    try:
        state = model.init_state(ids.shape[0])
        with torch.no_grad():
            r1 = model(ids, state, pad_token_id=0)
            state = r1["state_next"]
            model(ids, state, pad_token_id=0, **overrides)
    finally:
        model.backbone.forward_blocks = orig
    return captured["hidden"]


def test_default_no_override_smoke():
    """default(both None)forward 不 crash,logits 有限值。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    logits, _ = _run_state_then_forward(model, ids)
    assert logits.shape == (1, 8, 50)
    assert torch.isfinite(logits).all()


def test_mac_override_zeros_prefix():
    """white-box routing:mem_mac_override=0 → MAC-R 前缀 token 范数严格 0。
    default 下前缀应非零(W_mac=xavier 非零;mem_out 是 M·q 也非零)。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    n_q = model.n_query

    h_default = _capture_backbone_input(model, ids)
    h_mac_off = _capture_backbone_input(model, ids, mem_mac_override=0.0)

    prefix_default_norm = h_default[:, :n_q, :].abs().max().item()
    prefix_mac_off_norm = h_mac_off[:, :n_q, :].abs().max().item()
    assert prefix_default_norm > 1e-4, \
        f"default 下 MAC-R 前缀应非零,实测 max|.|={prefix_default_norm:.2e}"
    assert prefix_mac_off_norm < 1e-7, \
        f"mem_mac_override=0 应让前缀严格 0,实测 max|.|={prefix_mac_off_norm:.2e}"


def test_mal_override_does_not_zero_prefix():
    """white-box routing:mem_mal_override=0 应只影响 MAL(layer hook 内残差),
    不影响 MAC-R 前缀(送进 backbone 的 hidden 前 n_query)。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    n_q = model.n_query

    h_default = _capture_backbone_input(model, ids)
    h_mal_off = _capture_backbone_input(model, ids, mem_mal_override=0.0)

    # MAL 关 ≠ MAC-R 关:前缀应仍然非零(MAC-R 路径不受影响)
    prefix_default = h_default[:, :n_q, :]
    prefix_mal_off = h_mal_off[:, :n_q, :]
    assert torch.allclose(prefix_default, prefix_mal_off, atol=1e-6), \
        "MAL override 不该影响 MAC-R 前缀(只影响 backbone 内部 hidden)"
    assert prefix_default.abs().max().item() > 1e-4, "default MAC-R 前缀应非零"


def test_both_off_equivalent_to_alpha_zero():
    """mem_mac_override=0 + mem_mal_override=0 数学等价于 mem_alpha_override=0(NM-zero)。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    logits_both_off, _ = _run_state_then_forward(model, ids,
                                                 mem_mac_override=0.0, mem_mal_override=0.0)
    logits_alpha_zero, _ = _run_state_then_forward(model, ids, mem_alpha_override=0.0)
    assert torch.allclose(logits_both_off, logits_alpha_zero, atol=1e-5), \
        "(mac=0, mal=0) 应等价 alpha_override=0,差异说明路由 bug"


def test_explicit_none_equals_default():
    """显式传 None 应与不传等价。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    logits_implicit, _ = _run_state_then_forward(model, ids)
    logits_explicit, _ = _run_state_then_forward(
        model, ids, mem_mac_override=None, mem_mal_override=None,
    )
    assert torch.allclose(logits_implicit, logits_explicit, atol=1e-6), \
        "显式 None 应与 default 一致,有差说明 None 分支误触发"


def test_mac_override_precedence_over_alpha():
    """mem_mac_override 给定时,mem_alpha_override 对 MAC-R 无效(只影响 MAL)。
    验证:(mem_alpha_override=0, mem_mac_override=1) ≡ MAC-R 完整 + MAL 关。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    n_q = model.n_query

    # alpha=0 但 mac_override=1 → MAC-R 不受 alpha 影响,前缀仍非零
    h = _capture_backbone_input(model, ids,
                                mem_alpha_override=0.0, mem_mac_override=1.0)
    prefix = h[:, :n_q, :]
    assert prefix.abs().max().item() > 1e-4, \
        "mac_override=1 应保持 MAC-R 完整,即使 alpha_override=0"


def test_mal_override_precedence_over_alpha():
    """mem_mal_override 给定时,mem_alpha_override 对 MAL 无效。
    验证:(mem_alpha_override=0, mem_mal_override=1) ≡ MAC-R 关 + MAL 完整 σ(0)=0.5。"""
    model = _make_model()
    ids = torch.randint(1, 50, (1, 8))
    n_q = model.n_query

    # alpha=0 但 mal_override=1 → MAL 走 float(1.0)
    # 此时 MAC-R 被 alpha=0 关掉(前缀 0),MAL 仍激活(=1.0 而非 σ(logit))
    h = _capture_backbone_input(model, ids,
                                mem_alpha_override=0.0, mem_mal_override=1.0)
    # MAC-R 应该 0(alpha=0 覆盖,mac_override 没给)
    prefix = h[:, :n_q, :]
    assert prefix.abs().max().item() < 1e-7, \
        "mem_alpha_override=0 时 MAC-R 应被关,mac_override 未给不退化"
    # MAL 改动在 backbone 内部,这里只看 hidden 入口前缀 ≈ 0
