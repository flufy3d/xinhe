"""
StatePlugin — 可插拔的持久状态机制

核心设计:
- 状态实现为额外的 token，拼接在输入序列前面
- 复用 transformer 的 self-attention 作为读写机制
- 双层 Gate: 静态偏置 (脑区分化) + 动态投影 (内容选择)
- 渐进 scale: 从 0 开始，不破坏预训练模型

不依赖具体 backbone，将来换模型只需替换 backbone。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StatePlugin(nn.Module):
    """
    持久状态管理器。

    参数:
        n_state: 状态 token 数量
        state_dim: 状态维度 (应等于 backbone 的 hidden_size)
        state_scale_init: 渐进影响力初始值 (sigmoid 前)
        gate_bias_init: gate 静态偏置初始值
    """

    def __init__(
        self,
        n_state: int = 32,
        state_dim: int = 768,
        state_scale_init: float = -5.0,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.n_state = n_state
        self.state_dim = state_dim

        # 初始空白状态 (可学习参数，近零初始化)
        self.state_emb = nn.Parameter(torch.randn(n_state, state_dim) * 0.01)

        # 状态位置编码 (固定位置 0..n_state-1，不随 segment 变化)
        self.state_pos = nn.Embedding(n_state, state_dim)

        # 渐进影响力: sigmoid(state_scale_init) ≈ 0 时状态无影响
        self.state_scale = nn.Parameter(torch.tensor(state_scale_init))

        # --- 双层 Gate ---
        # 静态偏置: 维度天生的快慢倾向 (类似脑区分化)
        self.gate_bias = nn.Parameter(torch.full((n_state, state_dim), gate_bias_init))
        # 动态投影: 根据旧/新状态内容决定此刻该记还是该忘
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)

        # 输出投影: 将 transformer 输出的状态表示映射回状态空间
        # (可选，如果 backbone 的输出空间和状态空间一致则可用 identity)
        self.state_out_proj = nn.Linear(state_dim, state_dim, bias=False)
        nn.init.eye_(self.state_out_proj.weight)  # 初始化为恒等映射

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        创建空白初始状态。

        返回: (B, n_state, state_dim)
        """
        if device is None:
            device = self.state_emb.device
        # 用可学习的 state_emb 作为初始状态，扩展到 batch 维度
        return self.state_emb.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def inject(
        self,
        state: torch.Tensor,
        content_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将状态 token 注入到内容嵌入前面。

        参数:
            state: (B, n_state, D) 当前持久状态
            content_emb: (B, T, D) 内容 token 的嵌入

        返回:
            hidden_states: (B, n_state + T, D) 拼接后的序列
            mask: (1, 1, n_state + T, n_state + T) attention mask
        """
        B, T, D = content_emb.shape

        # 渐进影响力: scale 同时作用于状态值和 attention mask
        scale = torch.sigmoid(self.state_scale)

        # 1) 状态值缩放 (数值稳定性)
        pos_ids = torch.arange(self.n_state, device=state.device)
        state_with_pos = (state + self.state_pos(pos_ids).unsqueeze(0)) * scale

        # 对齐 dtype (Qwen=bfloat16, StatePlugin=float32)
        state_with_pos = state_with_pos.to(dtype=content_emb.dtype)

        # 拼接: [状态 | 内容]
        hidden_states = torch.cat([state_with_pos, content_emb], dim=1)

        # 2) Attention mask: content→state 通过 attention bias 门控
        #    乘以 4 确保 bias 足够大 (layer norm 会放大 state token 的值)
        #    scale_init=-5 时 bias≈-20，content 几乎看不到 state
        #    scale→1 时 bias→0，content 正常看到 state
        mask = self.build_mask(self.n_state, T, device=content_emb.device, dtype=content_emb.dtype)
        mask = mask.clone()
        mask[:, :, self.n_state:, :self.n_state] = torch.log(scale + 1e-8) * 4

        return hidden_states, mask

    def extract_and_update(
        self,
        output: torch.Tensor,
        state_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 transformer 输出中提取新状态，通过双层 gate 更新。

        参数:
            output: (B, n_state + T, D) transformer 完整输出
            state_old: (B, n_state, D) 上一轮的状态

        返回:
            content_output: (B, T, D) 内容部分的输出
            state_next: (B, n_state, D) 更新后的状态
        """
        # 分离状态和内容
        state_raw = output[:, :self.n_state, :]      # (B, n_state, D)
        content_output = output[:, self.n_state:, :]  # (B, T, D)

        # 输出投影 (对齐 dtype: backbone 输出可能是 bfloat16，plugin 参数是 float32)
        state_new = self.state_out_proj(state_raw.to(self.state_out_proj.weight.dtype))

        # --- 双层 Gate ---
        # 拼接旧状态和新状态: 模型看到 "我有什么" 和 "来了什么新的"
        combined = torch.cat([state_old, state_new], dim=-1)  # (B, n_state, 2*D)
        dynamic_logit = self.gate_proj(combined)               # (B, n_state, D)

        # gate = sigmoid(静态偏置 + 动态调制)
        gate = torch.sigmoid(self.gate_bias.unsqueeze(0) + dynamic_logit)

        # 更新: gate 高 → 保留旧状态; gate 低 → 采纳新状态
        state_next = gate * state_old + (1 - gate) * state_new

        return content_output, state_next

    @staticmethod
    def build_mask(
        n_state: int,
        n_content: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        构建 attention mask:
          - 状态→状态: 全可见 (双向)
          - 状态→内容: 全可见 (状态吸收所有内容)
          - 内容→状态: 全可见 (内容可读取所有状态)
          - 内容→内容: 因果遮蔽

        返回: (1, 1, S+T, S+T)，0=可见, -inf=遮蔽
        """
        total = n_state + n_content

        # 初始化为全遮蔽
        mask = torch.full((total, total), float("-inf"), device=device, dtype=dtype)

        # 状态→状态: 全可见
        mask[:n_state, :n_state] = 0

        # 状态→内容: 全可见 (状态可以看到所有内容)
        mask[:n_state, n_state:] = 0

        # 内容→状态: 全可见 (内容可以读取状态)
        mask[n_state:, :n_state] = 0

        # 内容→内容: 因果遮蔽 (下三角)
        causal = torch.triu(
            torch.full((n_content, n_content), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        mask[n_state:, n_state:] = causal

        # 添加 batch 和 head 维度: (1, 1, S+T, S+T)
        return mask.unsqueeze(0).unsqueeze(0)

    def get_state_stats(self, state: torch.Tensor) -> dict:
        """
        获取状态分析统计信息 (用于 /stats 命令)。

        参数:
            state: (B, n_state, D) 或 (n_state, D)

        返回:
            dict: 包含 gate_bias 分布、状态范数、有效秩等
        """
        if state.dim() == 3:
            state = state[0]  # 取第一个 batch

        gate_values = torch.sigmoid(self.gate_bias).detach()

        # 统计快/慢区域
        slow_mask = gate_values.mean(dim=-1) > 0.7  # 慢区: 平均 gate > 0.7
        fast_mask = gate_values.mean(dim=-1) < 0.3  # 快区: 平均 gate < 0.3

        # 状态有效秩 (通过 SVD)
        U, S, V = torch.linalg.svd(state.float())
        # 有效秩 = exp(entropy of normalized singular values)
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        effective_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm))).item()

        return {
            "scale": torch.sigmoid(self.state_scale).item(),
            "gate_mean": gate_values.mean().item(),
            "gate_std": gate_values.std().item(),
            "slow_dims": slow_mask.sum().item(),
            "fast_dims": fast_mask.sum().item(),
            "state_norm": state.norm().item(),
            "effective_rank": effective_rank,
        }
