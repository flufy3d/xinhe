"""
测试 XinheModel 整体前向传播 (v9: 双 NeuralMemoryPair, per full-attn 层)

使用 mock backbone 替代真实模型,避免依赖预训练权重。
"""
import torch
import pytest
import torch.nn as nn

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.xinhe_model import XinheModel
from xinhe.model.neural_memory_pair import XinheMemoryState


N_LAYERS = 4
HOOK_LAYERS = [1, 3]   # 模拟 full-attn 层在 layer 1 和 layer 3


class MockBackbone(nn.Module, BackboneBase):
    """用于测试的 mock backbone。layer 1 和 layer 3 模拟 full-attn 层(挂 NeuralMemoryPair)。"""

    def __init__(self, hidden_size=64, vocab_size=100):
        nn.Module.__init__(self)
        self._hidden_size = hidden_size
        self.embed_layer = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed(self, input_ids):
        return self.embed_layer(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None, position_ids=None, layer_hook=None):
        if layer_hook is not None:
            for i in range(N_LAYERS):
                hidden_states = layer_hook(hidden_states, i)
        out = self.linear(hidden_states)
        # 简单"伪 attention" mix:把序列 mean 加到每个位置(让 fresh_mem 段修改能影响 real 段输出),
        # 不然 mock 的 nn.Linear 是 per-position,test_mem_alpha_override_propagates 没法验
        out = out + hidden_states.mean(dim=1, keepdim=True) * 0.1
        return out

    def get_lm_head(self):
        return self._lm_head

    def get_hidden_size(self):
        return self._hidden_size

    def get_num_layers(self):
        return N_LAYERS

    def get_hook_layer_indices(self):
        return HOOK_LAYERS


def _build_model(d_total_eq_hidden=True):
    """构造测试模型。
    d_total_eq_hidden=True:n_heads*head_dim == hidden_size,_d_total 投影是 Identity
    d_total_eq_hidden=False:n_heads*head_dim ≠ hidden_size,有 Linear 投影"""
    if d_total_eq_hidden:
        n_heads, head_dim = 4, 16   # d_total = 64 = hidden_size
    else:
        n_heads, head_dim = 4, 8    # d_total = 32 ≠ hidden_size
    config = XinheConfig(
        hidden_size=64,
        n_heads=n_heads,
        head_dim=head_dim,
        mem_chunk_size=4,
        freeze_backbone=False,
        per_segment_checkpoint=False,
        phase="P-cap",
    )
    backbone = MockBackbone(hidden_size=64, vocab_size=100)
    return XinheModel(config, backbone=backbone)


@pytest.fixture
def model():
    return _build_model()


def test_forward_shape(model):
    """forward 输出形状正确 (v9: state 是 XinheMemoryState)"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state)

    assert result["logits"].shape == (B, T, 100)
    assert isinstance(result["state_next"], XinheMemoryState)
    assert set(result["state_next"].keys()) == set(HOOK_LAYERS)


def test_forward_with_labels(model):
    """提供 labels 时返回 loss(包含 gate entropy reg)"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state, labels=labels)

    assert "loss" in result
    assert "aux_loss" in result
    assert result["loss"].dim() == 0


def test_state_persistence(model):
    """状态在多个 segment 间传递(v9: per-layer NeuralMemState 演化)"""
    B, T = 1, 8
    state = model.init_state(B)

    seen_lu_first = None
    for step in range(3):
        input_ids = torch.randint(0, 100, (B, T))
        result = model(input_ids, state)
        state = result["state_next"]
        # 取第一层 hippo 的 past_last_update,验证它在演化
        layer_state = state[HOOK_LAYERS[0]]
        if layer_state.hippo is not None and layer_state.hippo.states is not None:
            lu = next(iter(layer_state.hippo.states[0].values())).clone()
            if seen_lu_first is None:
                seen_lu_first = lu
            elif step > 0:
                # state 跨 forward 应有变化
                assert not torch.allclose(lu, seen_lu_first, atol=1e-6)


def test_gradient_flow(model):
    """梯度能通过 NeuralMemoryPair 反向传播"""
    B, T = 1, 8
    state = model.init_state(B)

    input_ids_1 = torch.randint(0, 100, (B, T))
    result_1 = model(input_ids_1, state)
    state_1 = result_1["state_next"]

    input_ids_2 = torch.randint(0, 100, (B, T))
    labels_2 = torch.randint(0, 100, (B, T))
    result_2 = model(input_ids_2, state_1, labels=labels_2)

    loss = result_2["loss"]
    loss.backward()

    # NeuralMemoryPair 关键参数应该有 grad
    # pure MAC:alpha_logit 不在 loss 路径(x_out 被丢弃,只用 mem_out 填 fresh_mem)
    # 所以 alpha_logit.grad 是 None,不再 assert。gate_q + mem_token_init 才是关键
    pair = next(iter(model.memory.values()))
    assert pair.gate_q.weight.grad is not None
    if model.mem_token_init is not None:
        assert model.mem_token_init.grad is not None


def test_generate(model):
    """generate_with_state 能生成 token,state 类型保持"""
    B = 1
    state = model.init_state(B)
    input_ids = torch.randint(0, 100, (B, 4))

    with torch.no_grad():
        generated, new_state = model.generate_with_state(
            input_ids, state, max_new_tokens=8, temperature=1.0, top_p=1.0,
        )

    assert generated.shape[1] > input_ids.shape[1]
    assert isinstance(new_state, XinheMemoryState)


def test_trainable_params(model):
    """可训练参数数量 > 0(memory + gate_q + alpha 等)"""
    params = model.get_trainable_params()
    assert len(params) > 0
    count = model.get_trainable_param_count()
    assert count > 0


def test_forward_with_decoupled_dims():
    """n_heads*head_dim != hidden_size 时全流程正确(走 _d_total 投影)"""
    model = _build_model(d_total_eq_hidden=False)

    B, T = 2, 16
    state = model.init_state(B)
    assert isinstance(state, XinheMemoryState)
    assert set(state.keys()) == set(HOOK_LAYERS)

    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    result = model(input_ids, state, labels=labels)

    assert result["logits"].shape == (B, T, 100)
    assert isinstance(result["state_next"], XinheMemoryState)

    # 反传不报错
    result["loss"].backward()
    pair = next(iter(model.memory.values()))
    assert pair.gate_q.weight.grad is not None


def test_mem_alpha_override_propagates(model):
    """forward(mem_alpha_override=0.0) 透传到 layer hook → fresh_mem 位置不被注入。

    v9.5:Mock backbone 在 layer hook 处加 mem_out delta,override=0 时 delta=0;
    通过对比 logits 差异验证 override 透传(mock 没 attention,delta 直接进 hidden)。
    """
    B, T = 1, 8
    state = model.init_state(B)
    input_ids = torch.randint(0, 100, (B, T))
    result_with_mem = model(input_ids, state, mem_alpha_override=None)
    result_clean = model(input_ids, state, mem_alpha_override=0.0)
    # mem_token_init 启用时,fresh_mem 末层 hidden 必差异(MAC 注入 vs 干净路径)
    # mock backbone 的 forward_blocks 把 hidden 传过 layer_hook 后 linear 变换,
    # fresh_mem 位置的 logits(注入)与 real 位置的 logits 都会有差(简化 mock 全位置走同样 linear)
    if model.n_mem_tokens > 0:
        assert not torch.allclose(result_with_mem["logits"], result_clean["logits"], atol=1e-3)
