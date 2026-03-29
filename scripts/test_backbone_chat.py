"""
验证 Backbone 裸跑聊天 — 不加 StatePlugin，直接用预训练权重生成。
"""
import sys
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Windows 控制台 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def test_backbone(model_path: str, name: str):
    print(f"\n{'='*60}")
    print(f"  测试: {name} -- {model_path}")
    print(f"{'='*60}")

    # 加载
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if "qwen3-0.6b" in model_path else torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"设备: {device}, 参数: {sum(p.numel() for p in model.parameters()):,}")

    # 聊天格式
    prompts = [
        "你好，请介绍一下你自己。",
        "天空为什么是蓝色的？",
        "1+1等于几？",
    ]

    for prompt in prompts:
        print(f"\n用户: {prompt}")

        # 构造输入 — 关闭 think 模式
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 只取新生成部分
        new_ids = outputs[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"助手: {response}")

    print(f"\n{name} 测试完成!")


if __name__ == "__main__":
    # MiniMind 64M
    test_backbone("./models/minimind", "MiniMind 64M")

    # Qwen3-0.6B
    test_backbone("./models/qwen3-0.6b", "Qwen3-0.6B")
