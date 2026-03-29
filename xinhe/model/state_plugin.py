"""
StatePlugin — 可插拔的持久状态机制 (读写分离架构)

核心设计:
- 读状态 (Read-State) 放在序列开头: 携带上一轮的记忆，content 通过因果 attention 读取
- 写状态 (Write-State) 放在序列末尾: 通过因果 attention 吸收当前 segment 全部信息
- 标准因果 attention 自然防止信息泄漏: content 只能从旧 state 读，不能偷看当前答案
- 双层 Gate: 静态偏置 (脑区分化) + 动态投影 (内容选择)
- 渐进 scale: 控制 state token 的影响力

序列结构: [Read-State(旧) | Content | Write-State(新)]
- Read-State 只看到自己 → 携带旧记忆
- Content 看到 Read-State + 之前的 Content → 记忆融入思考
- Write-State 看到所有 → 吸收当前 segment 信息供下一轮使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StatePlugin(nn.Module):
    """
    持久状态管理器 (读写分离)。

    参数:
        n_state: 状态 token 数量 (读和写各 n_state 个)
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

        # 读状态位置编码 (序列开头，近零初始化)
        self.read_pos = nn.Embedding(n_state, state_dim)
        nn.init.normal_(self.read_pos.weight, std=0.01)

        # 写状态: 可学习查询嵌入 + 位置编码 (序列末尾)
        self.write_emb = nn.Parameter(torch.randn(n_state, state_dim) * 0.01)
        self.write_pos = nn.Embedding(n_state, state_dim)
        nn.init.normal_(self.write_pos.weight, std=0.01)

        # 渐进影响力: sigmoid(state_scale) 控制 state token 的值缩放
        self.state_scale = nn.Parameter(torch.tensor(state_scale_init))

        # --- 双层 Gate ---
        # 静态偏置: 维度天生的快慢倾向 (类似脑区分化)
        self.gate_bias = nn.Parameter(torch.full((n_state, state_dim), gate_bias_init))
        # 动态投影: 根据旧/新状态内容决定此刻该记还是该忘
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)

        # 输出投影: 将 transformer 输出的写状态映射回状态空间
        self.state_out_proj = nn.Linear(state_dim, state_dim, bias=False)
        nn.init.eye_(self.state_out_proj.weight)  # 初始化为恒等映射

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """创建空白初始状态。返回: (B, n_state, state_dim)"""
        if device is None:
            device = self.state_emb.device
        return self.state_emb.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def inject(
        self,
        state: torch.Tensor,
        content_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        构建 [Read-State | Content | Write-State] 序列。

        参数:
            state: (B, n_state, D) 当前持久状态 (上一轮的输出)
            content_emb: (B, T, D) 内容 token 的嵌入

        返回:
            hidden_states: (B, n_state + T + n_state, D)
            mask: (1, 1, L, L) 标准因果 attention mask
        """
        B, T, D = content_emb.shape
        scale = torch.sigmoid(self.state_scale)
        pos_ids = torch.arange(self.n_state, device=state.device)

        # Read tokens: 旧 state + 读位置编码, 按 scale 缩放
        read_tokens = (state + self.read_pos(pos_ids).unsqueeze(0)) * scale
        read_tokens = read_tokens.to(dtype=content_emb.dtype)

        # Write tokens: 可学习查询 + 写位置编码, 按 scale 缩放
        write_tokens = (
            self.write_emb.unsqueeze(0).expand(B, -1, -1)
            + self.write_pos(pos_ids).unsqueeze(0)
        ) * scale
        write_tokens = write_tokens.to(dtype=content_emb.dtype)

        # [Read | Content | Write]
        hidden_states = torch.cat([read_tokens, content_emb, write_tokens], dim=1)

        # 标准因果 mask: 位置 i 只能看到 0..i
        total_len = hidden_states.shape[1]
        mask = torch.triu(
            torch.full(
                (total_len, total_len), float("-inf"),
                device=content_emb.device, dtype=content_emb.dtype,
            ),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        return hidden_states, mask

    def extract_and_update(
        self,
        output: torch.Tensor,
        state_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 transformer 输出中提取 content 和 write-state，通过 gate 更新状态。

        参数:
            output: (B, n_state + T + n_state, D) transformer 完整输出
            state_old: (B, n_state, D) 上一轮的状态

        返回:
            content_output: (B, T, D) 内容部分的输出
            state_next: (B, n_state, D) 更新后的状态
        """
        n = self.n_state

        # 分离三部分
        # read_output = output[:, :n, :]       # 不需要，读状态已完成使命
        content_output = output[:, n:-n, :]    # (B, T, D)
        write_raw = output[:, -n:, :]          # (B, n_state, D)

        # 输出投影 (对齐 dtype)
        state_new = self.state_out_proj(write_raw.to(self.state_out_proj.weight.dtype))

        # --- 双层 Gate ---
        combined = torch.cat([state_old, state_new], dim=-1)  # (B, n_state, 2*D)
        dynamic_logit = self.gate_proj(combined)               # (B, n_state, D)
        gate = torch.sigmoid(self.gate_bias.unsqueeze(0) + dynamic_logit)

        # 更新: gate 高 → 保留旧状态; gate 低 → 采纳新状态
        state_next = gate * state_old + (1 - gate) * state_new

        return content_output, state_next

    def get_state_stats(self, state: torch.Tensor) -> dict:
        """
        获取状态分析统计信息 (用于 /stats 命令)。

        参数:
            state: (B, n_state, D) 或 (n_state, D)

        返回:
            dict: 包含 gate_bias 分布、状态范数、有效秩等
        """
        if state.dim() == 3:
            state = state[0]

        gate_values = torch.sigmoid(self.gate_bias).detach()

        slow_mask = gate_values.mean(dim=-1) > 0.7
        fast_mask = gate_values.mean(dim=-1) < 0.3

        U, S, V = torch.linalg.svd(state.float())
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
