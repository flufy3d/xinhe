"""
评估指标

核心指标:
1. retention_test — 跨轮次记忆保留率
2. update_test — 信息覆写能力
3. wipe_degradation — 状态清除后性能下降
4. timescale_distribution — 快/慢变量分布
"""
import torch
import numpy as np
from typing import Optional


def retention_test(
    model,
    tokenizer,
    distances: list[int] = [1, 2, 4, 8],
    num_trials: int = 50,
    device: torch.device = None,
) -> dict[int, float]:
    """
    记忆保留测试: 在第 0 轮告知信息，在第 d 轮查询，测准确率。

    返回: {distance: accuracy}
    """
    if device is None:
        device = next(model.parameters()).device

    names = ["小明", "小红", "张三", "李四", "王五"]
    fillers = [
        "今天天气怎么样？",
        "给我讲个故事。",
        "你觉得人工智能好不好？",
        "说一个有趣的事情。",
    ]

    results = {}
    model.eval()

    for d in distances:
        correct = 0

        for trial in range(num_trials):
            name = names[trial % len(names)]
            state = model.init_state(1).to(device)

            # 第 0 轮: 告知信息
            text = f"<s>用户：我叫{name}。\n助手：好的，{name}，我记住了。</s>"
            ids = tokenizer.encode(text, add_special_tokens=False)
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                result = model(input_tensor, state)
                state = result["state_next"]

            # 中间填充 d-1 轮闲聊
            for i in range(d - 1):
                filler = fillers[i % len(fillers)]
                text = f"<s>用户：{filler}\n助手：这是一个好问题。</s>"
                ids = tokenizer.encode(text, add_special_tokens=False)
                input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    result = model(input_tensor, state)
                    state = result["state_next"]

            # 第 d 轮: 查询
            query = f"<s>用户：我叫什么名字？\n助手："
            ids = tokenizer.encode(query, add_special_tokens=False)
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

            with torch.no_grad():
                generated, _ = model.generate_with_state(
                    input_tensor, state, max_new_tokens=32, temperature=0.1,
                )

            response = tokenizer.decode(generated[0, len(ids):].tolist(), skip_special_tokens=True)
            if name in response:
                correct += 1

        results[d] = correct / num_trials

    return results


def wipe_degradation(
    model,
    tokenizer,
    num_trials: int = 50,
    device: torch.device = None,
) -> dict[str, float]:
    """
    状态清除对比: 正常 vs 清除状态的记忆准确率。

    返回: {"with_state": accuracy, "without_state": accuracy, "degradation": delta}
    """
    if device is None:
        device = next(model.parameters()).device

    names = ["小明", "小红", "张三", "李四", "王五"]
    model.eval()

    with_state_correct = 0
    without_state_correct = 0

    for trial in range(num_trials):
        name = names[trial % len(names)]
        state = model.init_state(1).to(device)

        # 告知信息
        text = f"<s>用户：我叫{name}。\n助手：好的，{name}。</s>"
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model(input_tensor, state)
            state_after = result["state_next"]

        # 查询 (有状态)
        query = f"<s>用户：我叫什么？\n助手："
        q_ids = tokenizer.encode(query, add_special_tokens=False)
        q_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen, _ = model.generate_with_state(q_tensor, state_after, max_new_tokens=32, temperature=0.1)
        resp = tokenizer.decode(gen[0, len(q_ids):].tolist(), skip_special_tokens=True)
        if name in resp:
            with_state_correct += 1

        # 查询 (清除状态)
        blank_state = model.init_state(1).to(device)
        with torch.no_grad():
            gen, _ = model.generate_with_state(q_tensor, blank_state, max_new_tokens=32, temperature=0.1)
        resp = tokenizer.decode(gen[0, len(q_ids):].tolist(), skip_special_tokens=True)
        if name in resp:
            without_state_correct += 1

    with_acc = with_state_correct / num_trials
    without_acc = without_state_correct / num_trials

    return {
        "with_state": with_acc,
        "without_state": without_acc,
        "degradation": with_acc - without_acc,
    }


def timescale_distribution(state_history: list[torch.Tensor]) -> dict:
    """
    分析状态维度的时间尺度分布。

    参数:
        state_history: 每个 segment 后的状态列表, 每个 (n_state, D)

    返回:
        dict: 每个状态 token 的有效时间常数 + 统计信息
    """
    if len(state_history) < 3:
        return {"error": "需要至少 3 个时间步的状态历史"}

    # 转为 numpy: (T, n_state, D)
    states = torch.stack(state_history).detach().cpu().numpy()
    T, n_state, D = states.shape

    # 对每个 (state_token, dim) 计算自相关
    # 简化: 对每个 state token 取平均维度的自相关
    tau_per_token = []

    for s in range(n_state):
        # 这个 state token 在所有时间步的值: (T, D)
        trajectory = states[:, s, :]
        # 平均维度上的自相关
        mean_autocorr = _mean_autocorrelation(trajectory)
        # 拟合指数衰减获取时间常数
        tau = _fit_time_constant(mean_autocorr)
        tau_per_token.append(tau)

    tau_array = np.array(tau_per_token)

    return {
        "tau_per_token": tau_array.tolist(),
        "tau_mean": float(tau_array.mean()),
        "tau_std": float(tau_array.std()),
        "tau_min": float(tau_array.min()),
        "tau_max": float(tau_array.max()),
        "slow_count": int((tau_array > tau_array.mean() + tau_array.std()).sum()),
        "fast_count": int((tau_array < tau_array.mean() - tau_array.std()).sum()),
    }


def _mean_autocorrelation(trajectory: np.ndarray, max_lag: int = None) -> np.ndarray:
    """计算轨迹各维度自相关的均值"""
    T, D = trajectory.shape
    if max_lag is None:
        max_lag = min(T // 2, 20)

    # 中心化
    mean = trajectory.mean(axis=0, keepdims=True)
    centered = trajectory - mean

    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= T:
            break
        corr_per_dim = np.sum(centered[:T-lag] * centered[lag:], axis=0) / max(T - lag, 1)
        var = np.sum(centered ** 2, axis=0) / T
        var = np.maximum(var, 1e-10)
        autocorr[lag] = np.mean(corr_per_dim / var)

    return autocorr


def _fit_time_constant(autocorr: np.ndarray) -> float:
    """从自相关曲线拟合指数衰减的时间常数"""
    # autocorr ≈ exp(-t/tau) → log(autocorr) ≈ -t/tau
    # 简单做法: 找到自相关降到 1/e 的位置
    threshold = 1.0 / np.e
    for t, val in enumerate(autocorr):
        if val < threshold:
            return float(t)
    return float(len(autocorr))  # 如果从未降到 1/e，返回最大 lag
