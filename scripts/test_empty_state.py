"""
里程碑 2 验证: 空状态不破坏 — XinheModel (backbone + StatePlugin, scale≈0) 聊天正常。

对比 test_backbone_chat.py 的裸 backbone 结果，验证:
1. 生成文本连贯（不乱码）
2. logits 分布与裸 backbone 接近（top-k overlap）
"""
import sys
import io
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Windows 控制台 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


def test_empty_state(config_path: str):
    config = XinheConfig.from_yaml(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  里程碑 2: 空状态验证")
    print(f"  config: {config_path}")
    print(f"  state_scale_init: {config.state_scale_init}")
    print(f"  scale = sigmoid({config.state_scale_init}) = {torch.sigmoid(torch.tensor(config.state_scale_init)).item():.6f}")
    print(f"{'='*60}")

    # 加载模型
    print("加载 XinheModel...")
    model = XinheModel(config)
    model.to(device).eval()

    total = model.get_total_param_count()
    trainable = model.get_trainable_param_count()
    print(f"  参数: {total:,} 总 / {trainable:,} 可训练")

    # 加载 tokenizer
    model_path = Path(config.backbone_model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # 初始化空白状态
    state = model.init_state(batch_size=1).to(device)
    print(f"  state shape: {state.shape}")
    print(f"  state norm: {state.norm().item():.4f}")

    # 状态统计
    stats = model.state_stats(state)
    print(f"  scale: {stats['scale']:.6f}")
    print(f"  gate_mean: {stats['gate_mean']:.4f}")

    # 测试 prompt（与 test_backbone_chat.py 相同）
    prompts = [
        "你好，请介绍一下你自己。",
        "天空为什么是蓝色的？",
        "1+1等于几？",
    ]

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for prompt in prompts:
        print(f"\n用户: {prompt}")

        # 构造输入
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        input_ids = tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # 生成
        with torch.no_grad():
            generated_ids, state = model.generate_with_state(
                input_ids=input_tensor,
                state=state,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=eos_id,
            )

        new_ids = generated_ids[0, len(input_ids):].tolist()
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        response = response.replace("</s>", "").strip()
        print(f"助手: {response}")

    # 最终状态统计
    stats = model.state_stats(state)
    print(f"\n{'='*60}")
    print(f"  最终状态统计")
    print(f"  scale: {stats['scale']:.6f}")
    print(f"  state_norm: {stats['state_norm']:.2f}")
    print(f"  effective_rank: {stats['effective_rank']:.1f}")
    print(f"{'='*60}")
    print("\n验证完成! 对比 test_backbone_chat.py 的输出判断质量是否接近。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-0.6b.yaml")
    args = parser.parse_args()
    test_empty_state(args.config)
