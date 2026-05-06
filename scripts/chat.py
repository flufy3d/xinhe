"""
交互式聊天验证工具 — 心核核心验证入口

命令:
    /save <name>     保存当前状态到文件
    /load <name>     加载之前保存的状态
    /wipe            清除状态 (对比实验: 清除后还记得吗?)
    /stats           显示状态分析 (gate分布、活跃维度、有效秩)
    /burnin <text>   用文本初始化 persona 状态
    /reset           重置为空白状态
    /quit            退出

用法:
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/xinhe_step_5000.pt
    python scripts/chat.py --state saved_states/james.pt
    python scripts/chat.py --no-stream       # 关闭流式逐字输出
"""
import argparse
import sys
from pathlib import Path

import torch

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import ensure_chat_template


STATES_DIR = Path("saved_states")


def load_tokenizer(config: XinheConfig):
    """加载 tokenizer"""
    model_path = Path(config.backbone_model_path).resolve()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)
    return tokenizer


def print_stats(model: XinheModel, state):
    """打印 NeuralMemoryPair 状态简要(per-layer alpha 平均)"""
    stats = model.state_stats(state)
    print(f"\n{'='*50}")
    print(f"  v9 Memory 状态 ({stats['n_layers']} 层 NeuralMemoryPair)")
    print(f"{'='*50}")
    print(f"  alpha_mean:        {stats['alpha_mean']:.4f}")
    print(f"{'='*50}\n")


def save_state(state, name: str):
    """保存状态到文件(v9 XinheMemoryState 用 torch.save 序列化整个对象)"""
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    path = STATES_DIR / f"{name}.pt"
    torch.save(state, path)
    print(f"  状态已保存到 {path}")


def load_state(name: str, device: torch.device):
    """从文件加载状态(v9 XinheMemoryState)。"""
    path = STATES_DIR / f"{name}.pt"
    if not path.exists():
        print(f"  错误: 找不到状态文件 {path}")
        if STATES_DIR.exists():
            files = list(STATES_DIR.glob("*.pt"))
            if files:
                print(f"  可用的状态: {', '.join(f.stem for f in files)}")
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    print(f"  状态已从 {path} 加载")
    return state


def main():
    parser = argparse.ArgumentParser(description="心核 交互式聊天")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--state", type=str, default=None, help="初始状态文件路径")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    # 加载 checkpoint（如提供）
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # 优先使用配置文件；当未显式传 --config 时，自动采用 checkpoint 中保存的配置
    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if checkpoint and not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")
    elif checkpoint and not config_explicit and "config" not in checkpoint:
        print("  提示: 请使用与 checkpoint 匹配的 --config。")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核 (Xinhe) 交互式聊天 ===")
    print(f"设备: {device}")
    print("加载模型...")

    model = XinheModel(config)

    # 加载训练好的 checkpoint
    if checkpoint:
        # 若用户显式指定了 --config，但与 checkpoint 配置不一致，给出提示
        ckpt_cfg = checkpoint.get("config")
        if config_explicit and isinstance(ckpt_cfg, XinheConfig):
            if (ckpt_cfg.backbone_type != config.backbone_type) or (ckpt_cfg.hidden_size != config.hidden_size):
                print("  警告: --config 与 checkpoint 不匹配。")
        if "memory_pair_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'memory_pair_state' 键。v9 不兼容 v8 hippocampus_state。"
            )
        model.memory.load_state_dict(checkpoint["memory_pair_state"], strict=True)
        print(f"  checkpoint 已加载: {args.checkpoint}")

    model.to(device)
    model.eval()

    # 加载 tokenizer
    tokenizer = load_tokenizer(config)

    # 初始化状态
    if args.state:
        state = torch.load(args.state, map_location=device, weights_only=True)
        print(f"  状态已加载: {args.state}")
    else:
        state = model.init_state(batch_size=1).to(device)
        print("  空白状态已初始化")

    trainable = model.get_trainable_param_count()
    total = model.get_total_param_count()
    print(f"  参数: {total:,} 总 / {trainable:,} 可训练")
    print()
    print("输入消息开始对话。输入 /help 查看命令。")
    print("-" * 50)

    # 对话历史 (用于当前 segment 的上下文)
    turn_count = 0

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # --- 处理命令 ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            cmd_arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print("再见！")
                break

            elif cmd == "/help":
                print("  /save <name>   — 保存当前状态")
                print("  /load <name>   — 加载状态")
                print("  /wipe          — 清除状态 (对比实验)")
                print("  /stats         — 状态分析")
                print("  /reset         — 重置为空白状态")
                print("  /burnin <text> — 用文本初始化 persona")
                print("  /quit          — 退出")

            elif cmd == "/save":
                name = cmd_arg or f"state_turn{turn_count}"
                save_state(state, name)

            elif cmd == "/load":
                if not cmd_arg:
                    print("  用法: /load <name>")
                else:
                    loaded = load_state(cmd_arg, device)
                    if loaded is not None:
                        state = loaded.to(device)

            elif cmd == "/wipe":
                state = model.init_state(batch_size=1).to(device)
                print("  状态已清除（重置为空白）")

            elif cmd == "/stats":
                print_stats(model, state)

            elif cmd == "/reset":
                state = model.init_state(batch_size=1).to(device)
                turn_count = 0
                print("  状态和对话已重置")

            elif cmd == "/burnin":
                if not cmd_arg:
                    print("  用法: /burnin <system prompt text>")
                else:
                    print(f"  Burn-in: '{cmd_arg}'")
                    token_ids = tokenizer.encode(cmd_arg, add_special_tokens=False)
                    seg_len = config.turn_max_tokens
                    segments = []
                    for i in range(0, len(token_ids), seg_len):
                        seg = token_ids[i:i+seg_len]
                        segments.append(torch.tensor(seg, dtype=torch.long, device=device))
                    with torch.no_grad():
                        state = model.burn_in(segments, batch_size=1)
                    print("  Burn-in 完成")
            else:
                print(f"  未知命令: {cmd}，输入 /help 查看帮助")

            continue

        # --- 正常对话 ---
        turn_count += 1

        # 构建输入 (使用统一的 ChatML 模板，与训练一致)
        messages = [{"role": "user", "content": user_input}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # eos token id
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stream = not args.no_stream

        # --- 流式输出 ---
        if stream:
            print(f"\n心核: ", end="", flush=True)
            token_buf = []       # 累积 token id，用于增量解码
            printed_len = 0      # 已打印的字符数
            skip_ids = {tokenizer.convert_tokens_to_ids(t)
                        for t in ["</s>", "<|im_end|>", "<|endoftext|>"]} - {None}

            def _stream_token(token_id: int):
                nonlocal printed_len
                if token_id in skip_ids:
                    return
                token_buf.append(token_id)
                text = tokenizer.decode(token_buf, skip_special_tokens=False)
                new_text = text[printed_len:]
                if not new_text:
                    return
                print(new_text, end="", flush=True)
                printed_len = len(text)

            with torch.no_grad():
                generated_ids, state = model.generate_with_state(
                    input_ids=input_tensor,
                    state=state,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=eos_id,
                    token_callback=_stream_token,
                )
            print()  # 换行
        else:
            # --- 非流式输出 ---
            with torch.no_grad():
                generated_ids, state = model.generate_with_state(
                    input_ids=input_tensor,
                    state=state,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=eos_id,
                )
            new_ids = generated_ids[0, len(input_ids):].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=False)
            for tag in ["</s>", "<|im_end|>", "<|endoftext|>"]:
                response = response.replace(tag, "")
            response = response.strip()
            print(f"\n心核: {response}")

        alphas = [torch.sigmoid(p.alpha_logit).item() for p in model.memory.values()]
        alpha_mean = sum(alphas) / max(len(alphas), 1)
        print(f"  [轮次 {turn_count} | alpha={alpha_mean:.3f}]")



if __name__ == "__main__":
    main()
