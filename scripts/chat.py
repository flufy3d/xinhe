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


def print_stats(model: XinheModel, state: torch.Tensor):
    """打印状态分析"""
    stats = model.state_stats(state)
    print(f"\n{'='*50}")
    print(f"  状态分析")
    print(f"{'='*50}")
    print(f"  scale (影响力):  {stats['scale']:.4f}")
    print(f"  gate 均值:       {stats['gate_mean']:.4f}")
    print(f"  gate 标准差:     {stats['gate_std']:.4f}")
    print(f"  慢区维度 (>0.7): {stats['slow_dims']:.0f} / {model.config.n_state}")
    print(f"  快区维度 (<0.3): {stats['fast_dims']:.0f} / {model.config.n_state}")
    print(f"  状态范数:        {stats['state_norm']:.2f}")
    print(f"  有效秩:          {stats['effective_rank']:.1f}")
    print(f"{'='*50}\n")


def save_state(state: torch.Tensor, name: str):
    """保存状态到文件"""
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    path = STATES_DIR / f"{name}.pt"
    torch.save(state.cpu(), path)
    print(f"  状态已保存到 {path}")


def load_state(name: str, device: torch.device) -> torch.Tensor:
    """从文件加载状态"""
    path = STATES_DIR / f"{name}.pt"
    if not path.exists():
        print(f"  错误: 找不到状态文件 {path}")
        # 列出可用的状态
        if STATES_DIR.exists():
            files = list(STATES_DIR.glob("*.pt"))
            if files:
                print(f"  可用的状态: {', '.join(f.stem for f in files)}")
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    print(f"  状态已从 {path} 加载")
    return state


def main():
    parser = argparse.ArgumentParser(description="心核 交互式聊天")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--state", type=str, default=None, help="初始状态文件路径")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    # 加载配置和模型
    config = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核 (Xinhe) 交互式聊天 ===")
    print(f"设备: {device}")
    print("加载模型...")

    model = XinheModel(config)

    # 加载训练好的 checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.plugin.load_state_dict(checkpoint["plugin_state"])
        # 恢复 LoRA
        from xinhe.model.lora import LoRALinear
        lora_state = checkpoint.get("lora_state", {})
        for name, module in model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state:
                    module.lora_B.data = lora_state[f"{name}.lora_B"]
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
                    seg_len = config.segment_length
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

        # 生成回复
        with torch.no_grad():
            generated_ids, state = model.generate_with_state(
                input_ids=input_tensor,
                state=state,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_id,
            )

        # 解码输出 (只取新生成的部分)
        new_ids = generated_ids[0, len(input_ids):].tolist()
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        # 去掉可能的结束标记
        response = response.replace("</s>", "").strip()

        print(f"\n心核: {response}")
        print(f"  [轮次 {turn_count} | scale={torch.sigmoid(model.plugin.state_scale).item():.3f}]")



if __name__ == "__main__":
    main()
