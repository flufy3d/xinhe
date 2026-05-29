"""QueryHead — learned read-query 生成器(单全局记忆)。

q 必须来自 embedding / backbone 低层 h_pre,**不能来自 last layer**:
MAC-R 把 mem_out 投成 token 拼进*输入序列*,若 q 依赖完整 forward 输出会循环依赖。

训练信号脆弱性(M 对 outer backprop detach):QueryHead 学的是"统计上哪种 q
最可能 retrieve 到 target",必须配 q 多样性监控(cosine_diversity ≥ 0.3)防 mode collapse。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_memory_pair import AdaptiveRMSNorm  # autocast-safe RMSNorm(γ 保 fp32)


class QueryHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, n_query: int = 16):
        super().__init__()
        self.d_model = int(d_model)
        self.d_key = int(d_key)
        self.n_query = int(n_query)
        self.norm = AdaptiveRMSNorm(d_model)
        self.proj = nn.Linear(d_model, self.d_key * self.n_query, bias=False)
        # 非零 init:零 init 会让每个 q 恒等 → 第一步就 mode collapse(这正是要防的)。
        # frozen-backbone 安全由下游 W_mac/W_mal 的零 init 保证,不靠 QueryHead 零 init。
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """h_last: (B, d_model) —— caller 传 pad-aware 的最后一个非 pad token 向量。
        返回 (B, n_query, d_key)。"""
        h = self.norm(h_last)
        return self.proj(h).view(-1, self.n_query, self.d_key)

    @staticmethod
    def cosine_diversity(q: torch.Tensor) -> torch.Tensor:
        """一组 query 向量两两 (1 - cos) 均值;→1 多样,→0 = mode collapse(阈值 ≥ 0.3)。
        q: (..., d) —— 所有前缀维都当独立向量展平(可传跨 turn 的 q 测 turn 间塌缩,
        或单 forward 的 (B, n_query, d) 测整体多样性)。精确去对角,对零向量也鲁棒。"""
        v = q.reshape(-1, q.shape[-1]).float()
        n = v.shape[0]
        if n < 2:
            return torch.zeros((), device=q.device, dtype=torch.float32)
        vn = F.normalize(v, dim=-1)
        sim = vn @ vn.t()                                      # (n, n)
        off_mean = (sim.sum() - sim.diag().sum()) / (n * (n - 1))  # 去对角两两平均 cos
        return 1.0 - off_mean
