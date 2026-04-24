"""
状态可视化

生成:
1. 状态热力图 (维度 × 时间)
2. 状态 PCA 轨迹
3. 有效秩随时间变化
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def plot_state_heatmap(state_history: list[torch.Tensor], save_path: str = "figures/state_heatmap.png"):
    """
    绘制状态热力图 (状态 token × 时间步)

    state_history: 每个时间步的状态, (n_state, D)
    """
    if not state_history:
        print("  无状态历史数据")
        return

    # (T, n_state) — 取每个 token 的范数作为活跃度
    norms = torch.stack([s.norm(dim=-1) for s in state_history]).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(norms.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Segment 编号')
    ax.set_ylabel('状态 token 编号')
    ax.set_title('状态活跃度热力图 (范数)')
    plt.colorbar(im, ax=ax, label='范数')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  状态热力图已保存: {save_path}")


def plot_state_pca(state_history: list[torch.Tensor], save_path: str = "figures/state_pca.png"):
    """状态 PCA 轨迹 (2D)"""
    if len(state_history) < 3:
        print("  需要至少 3 个时间步")
        return

    # (T, n_state * D) — 展平
    flat = torch.stack([s.flatten() for s in state_history]).detach().cpu().numpy()

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(flat)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.linspace(0, 1, len(projected))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=colors, cmap='coolwarm', s=50)
    # 画轨迹线
    ax.plot(projected[:, 0], projected[:, 1], 'k-', alpha=0.3)
    # 标记起点和终点
    ax.scatter(*projected[0], color='green', s=200, marker='o', zorder=5, label='起点')
    ax.scatter(*projected[-1], color='red', s=200, marker='*', zorder=5, label='终点')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('状态 PCA 轨迹')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='时间步')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA 轨迹图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="心核 状态可视化")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    from xinhe.model.config import XinheConfig
    from xinhe.model.xinhe_model import XinheModel

    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = XinheModel(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "hippocampus_state" not in checkpoint:
        raise RuntimeError(
            "checkpoint 缺少 'hippocampus_state' 键。v7 不兼容 v5c/v6 旧格式。"
        )
    model.hippocampus.load_state_dict(checkpoint["hippocampus_state"], strict=True)
    model.to(device)
    model.eval()

    print("=== 心核 状态可视化 ===")

    # 生成状态历史 (用合成对话)
    print("  生成状态历史...")
    from transformers import AutoTokenizer
    model_path = Path(config.backbone_model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    state = model.init_state(1).to(device)
    state_history = [state[0].clone()]

    texts = [
        "我叫张三，我住在北京。",
        "今天天气很好。",
        "我喜欢吃火锅。",
        "给我讲一个故事。",
        "人工智能很有趣。",
        "我在做AI研究。",
        "推荐一本好书。",
        "明天要去上海出差。",
    ]

    for text in texts:
        full = f"<s>用户：{text}\n助手：好的。</s>"
        ids = tokenizer.encode(full, add_special_tokens=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model(input_tensor, state)
            state = result["state_next"]
        state_history.append(state[0].clone())

    # 热力图
    plot_state_heatmap(state_history, f"{args.output_dir}/state_heatmap.png")

    # PCA 轨迹
    plot_state_pca(state_history, f"{args.output_dir}/state_pca.png")

    print("可视化完成。")


if __name__ == "__main__":
    main()
