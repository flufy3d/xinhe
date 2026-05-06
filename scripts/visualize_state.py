"""状态可视化 — v9 deprecated。

v8 单 W (B,H,d_v,d_k) 张量的热力图 / PCA 在 v9 不再适用:
  - state 是 per-layer LayerMemState(NeuralMemState 命名元组),含 weights / states / momentum
  - 真正的"记忆活跃度"应该看 alpha + gate_entropy + Hippo/Neo weight norm

P0 阶段直接 raise,避免 stale 路径误导。等 P1+ 重写为 per-layer 可视化时再实现。
"""
import sys


def main():
    print(
        "[visualize_state.py] v8 状态可视化已 deprecated。\n"
        "v9 的 NeuralMemoryPair 状态结构跟 v8 单 W 不同,需重写。\n"
        "P1+ 将提供 per-layer alpha / gate / weight norm 可视化。",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
