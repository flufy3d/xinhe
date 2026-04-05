"""
Think 训练数据生成 (库 + 预览)

用干净 backbone (无 LoRA) 对包含 fact 的 prompt 生成 think 模式回复，
训练时 fact 从 text 移入 state，backbone 原始回复作为训练 target。

所有参数 (比例、混合配置等) 由 configs/curriculum.yaml 定义，
通过 generate_data.py 统一分发调用。

正式生成:
    python scripts/generate_data.py --config configs/curriculum_qwen.yaml --stage 14_think

预览 episode 结构:
    python scripts/generate_think_data.py --preview 3
"""
import argparse
import json
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

# 添加项目根目录和 scripts 目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from generate_memory_data import (
    sample_facts, FACT_TEMPLATES, RECALL_TEMPLATES, FILLERS,
    DYNAMIC_CONTENT_TEMPLATES, generate_dynamic_content,
    make_turn, episode_to_jsonl, generate_data,
)


# ── Think 提问模板 ──

# 总结类
SUMMARY_QUESTIONS = [
    "总结一下你对{e}的了解。",
    "你都知道{e}什么信息？",
    "说说你知道的关于{e}的事。",
    "你对{e}了解多少？",
    "把你知道关于{e}的事都说说。",
]

# 开放推理
REASON_QUESTIONS = [
    "根据你对{e}的了解，你觉得{e}是什么样的人？",
    "给{e}推荐点适合的活动。",
    "你觉得{e}的生活怎么样？",
    "根据你知道的，给{e}一些建议。",
    "你觉得{e}会喜欢什么样的礼物？",
]

# 组合推理 (两个 fact 类别搭配)
COMBINATION_QUESTIONS = {
    ("city", "hobby"): [
        "根据{e}住的地方和爱好，推荐个周末活动。",
        "{e}住的城市适合{e}的爱好吗？",
    ],
    ("age", "job"): [
        "你觉得{e}的年龄和职业匹配吗？",
        "以{e}的年龄做这个工作，你怎么看？",
    ],
    ("city", "food"): [
        "{e}住的地方能吃到{e}喜欢的食物吗？",
        "在{e}的城市，{e}爱吃的东西好找吗？",
    ],
    ("hobby", "age"): [
        "{e}这个年纪适合这个爱好吗？",
        "你觉得{e}的爱好和年龄搭吗？",
    ],
    ("job", "hobby"): [
        "{e}的工作和爱好冲突吗？",
        "{e}下班后的爱好和工作有关系吗？",
    ],
    ("name", "city"): [
        "说说{e}这个人和{e}住的地方。",
    ],
    ("food", "hobby"): [
        "{e}喜欢吃的东西和爱好搭吗？",
        "根据{e}的爱好推荐个配套的美食。",
    ],
    ("pet", "city"): [
        "{e}住的地方适合养{e}的宠物吗？",
        "在{e}的城市养宠物方便吗？",
    ],
    ("pet", "hobby"): [
        "{e}的宠物和{e}的爱好搭吗？",
    ],
    ("job", "city"): [
        "{e}的城市适合{e}的工作发展吗？",
    ],
    ("food", "age"): [
        "你觉得{e}这个年龄吃这些健康吗？",
    ],
    ("name", "job"): [
        "说说{e}这个人和{e}的工作。",
    ],
}

# 续聊短输入
CONTINUATION_INPUTS = [
    "嗯。", "然后呢？", "还有呢？", "真的吗？", "是吗？",
    "有意思。", "继续说。", "我知道了。", "哦。", "好吧。",
    "那怎么办？", "你觉得呢？", "还能说说吗？", "为什么？",
    "所以呢？", "接着呢？", "哈哈。", "原来如此。",
]

# 心跳引导 system prompt
HEARTBEAT_PROMPTS = [
    "你是一个有自己想法的存在。以下是你知道的事：\n{context}\n现在没有人对你说话。你可以选择说点什么，也可以沉默。如果你要说，说你真正想说的。",
    "你是一个有自己想法的存在。以下是你知道的事：\n{context}\n说说你现在的想法。",
    "你是一个有自己想法的存在。以下是你知道的事：\n{context}\n你注意到了什么？",
    "你是一个有自己想法的存在。以下是你知道的事：\n{context}\n有什么是你在意的事？",
    "你是一个有自己想法的存在。以下是你知道的事：\n{context}\n你想对谁说点什么？",
]

# 纯逻辑题
PURE_LOGIC_QUESTIONS = [
    # 算术
    "3加5等于几？",
    "12的一半是多少？",
    "一个苹果2元，买5个要多少钱？",
    "7乘以8等于多少？",
    "100减37等于多少？",
    "24除以6等于几？",
    "如果你有15块钱，花了9块，还剩多少？",
    "3个人平分12个苹果，每人几个？",
    "一打鸡蛋是多少个？",
    "一周有几天？",
    "25加37等于多少？",
    "99减46等于多少？",
    "8乘以9等于多少？",
    "144除以12等于多少？",
    "一年有多少个月？",
    # 比较
    "5和8哪个大？",
    "如果小明比小红高，小红比小华高，谁最矮？",
    "如果A比B重，C比A重，谁最轻？",
    "1米和100厘米，哪个长？",
    "如果甲跑得比乙快，丙跑得比甲快，谁跑得最慢？",
    # 三段论
    "所有猫都是动物，小花是猫，小花是动物吗？",
    "所有学生都要上学，小明是学生，小明要上学吗？",
    "所有鸟都有翅膀，企鹅是鸟，企鹅有翅膀吗？",
    "如果下雨就带伞，现在下雨了，要带伞吗？",
    "如果所有的苹果都是水果，有些水果是红色的，能确定有苹果是红色的吗？",
    # 排列组合
    "红黄蓝三种颜色能组成几种不同的两色搭配？",
    "ABC三个字母能排成几种不同的顺序？",
    "从5个人中选2个人，有几种选法？",
    # 逻辑推理
    "小明说的话总是假的。小明说「今天是周一」，今天是周一吗？",
    "小明排在小红前面，小华排在小明前面，谁排在最前面？",
    "今天不是周末，明天也不是周末，今天可能是星期几？",
    "有3个盒子，一个装苹果，一个装橙子，一个混装。标签全贴错了。从混装盒取一个是苹果，那混装盒实际装什么？",
]


# ── Episode 构造 ──

def build_fact_think_episode(
    rng: random.Random,
    num_facts: int = 3,
    question_type: str = "single",
    max_pre_filler: int = 2,
    min_distance: int = 1,
    max_distance: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    构造 fact 推理 think episode。

    返回:
        turns: 训练用轮次（tell + filler + question，最后一轮 assistant 待填充）
        messages: 给 backbone 的 messages（backbone 能看到完整 facts）
    """
    facts = sample_facts(rng, num_facts)
    turns = []
    messages = []

    # 前置闲聊
    pre_count = rng.randint(0, max_pre_filler)
    for _ in range(pre_count):
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})
        messages.append({"role": "user", "content": filler[0]})
        messages.append({"role": "assistant", "content": filler[1]})

    # 告知阶段
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        user_text = template[0].format(v=fact["value"])
        asst_text = template[1].format(v=fact["value"])
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": True})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": asst_text})

    # 闲聊填充
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})
        messages.append({"role": "user", "content": filler[0]})
        messages.append({"role": "assistant", "content": filler[1]})

    # 构造提问
    entity = "我"

    if question_type == "single":
        target_fact = rng.choice(facts)
        cat = target_fact["category"]
        template = rng.choice(RECALL_TEMPLATES[cat])
        question = template[0]

    elif question_type == "summary":
        question = rng.choice(SUMMARY_QUESTIONS).format(e=entity)

    elif question_type == "reason":
        question = rng.choice(REASON_QUESTIONS).format(e=entity)

    elif question_type == "combination":
        cats = [f["category"] for f in facts]
        available = []
        for combo in combinations(cats, 2):
            if combo in COMBINATION_QUESTIONS:
                available.append(combo)
            elif combo[::-1] in COMBINATION_QUESTIONS:
                available.append(combo[::-1])
        if available:
            key = rng.choice(available)
            question = rng.choice(COMBINATION_QUESTIONS[key]).format(e=entity)
        else:
            # fallback: 没有匹配的组合，用 reason
            question = rng.choice(REASON_QUESTIONS).format(e=entity)

    else:
        question = rng.choice(SUMMARY_QUESTIONS).format(e=entity)

    turns.append({"user": question, "assistant": "", "train_loss": True})
    messages.append({"role": "user", "content": question})

    return turns, messages


def build_continuation_episode(
    rng: random.Random,
    num_facts: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    构造续聊 think episode。

    前几轮: fact 告知 + 动态闲聊（写入 state 的能力已训练过）
    最后一轮: 短用户输入，backbone 看完整历史续聊
    """
    facts = sample_facts(rng, num_facts)
    turns = []
    messages = []

    # fact 告知
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        user_text = template[0].format(v=fact["value"])
        asst_text = template[1].format(v=fact["value"])
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": True})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": asst_text})

    # 动态对话 1-2 轮
    for _ in range(rng.randint(1, 2)):
        user_text, asst_text = generate_dynamic_content(rng)
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": False})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": asst_text})

    # 短续聊输入
    continuation = rng.choice(CONTINUATION_INPUTS)
    turns.append({"user": continuation, "assistant": "", "train_loss": True})
    messages.append({"role": "user", "content": continuation})

    return turns, messages


def build_heartbeat_episode(
    rng: random.Random,
    num_facts: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    构造心跳 think episode。

    前几轮: fact 告知 + 对话（写入 state）
    最后一轮: 空输入
    生成时: backbone 看 system prompt + 上下文
    训练时: system prompt 去掉，只有空输入 + think 回复
    """
    facts = sample_facts(rng, num_facts)
    turns = []

    # fact 告知
    # 心跳 context 用第三人称描述，避免 backbone 把用户 facts 当成自己的
    HEARTBEAT_CONTEXT = {
        "name": "用户叫{v}",
        "number": "用户的编号是{v}",
        "city": "用户住在{v}",
        "food": "用户喜欢吃{v}",
        "job": "用户的职业是{v}",
        "hobby": "用户喜欢{v}",
        "age": "用户{v}岁",
        "pet": "用户养了{v}",
    }
    fact_lines = []
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        user_text = template[0].format(v=fact["value"])
        asst_text = template[1].format(v=fact["value"])
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": True})
        context_text = HEARTBEAT_CONTEXT.get(cat, "用户提到{v}").format(v=fact["value"])
        fact_lines.append(context_text)

    # 1-2 轮动态对话
    chat_lines = []
    for _ in range(rng.randint(1, 2)):
        user_text, asst_text = generate_dynamic_content(rng)
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": False})
        chat_lines.append(f"用户: {user_text}\nAI: {asst_text}")

    # backbone 的 system prompt (含上下文，只在数据生成阶段存在)
    context = "\n".join(fact_lines)
    if chat_lines:
        context += "\n最近的对话：\n" + "\n".join(chat_lines)
    system_prompt = rng.choice(HEARTBEAT_PROMPTS).format(context=context)

    messages_for_backbone = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ""},
    ]

    # 训练: 空输入 + think 回复
    turns.append({"user": "", "assistant": "", "train_loss": True})

    return turns, messages_for_backbone


def build_pure_logic_episode(
    rng: random.Random,
    num_noise_facts: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    构造纯逻辑 think episode。

    前几轮: 随机无关 facts（制造噪声 state，教模型忽略无用信息）
    最后一轮: 纯逻辑题
    Backbone 只看到逻辑题，生成 think 推理
    """
    # 噪声 facts
    facts = sample_facts(rng, num_noise_facts)
    turns = []

    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        user_text = template[0].format(v=fact["value"])
        asst_text = template[1].format(v=fact["value"])
        turns.append({"user": user_text, "assistant": asst_text, "train_loss": True})

    # 1 轮 filler
    filler = rng.choice(FILLERS)
    turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 逻辑题
    question = rng.choice(PURE_LOGIC_QUESTIONS)
    turns.append({"user": question, "assistant": "", "train_loss": True})

    # backbone 只看逻辑题（干净 prompt，不含噪声 facts）
    messages_for_backbone = [{"role": "user", "content": question}]

    return turns, messages_for_backbone


# ── Backbone 推理 ──

def load_backbone(model_path: str, device: str = "cuda"):
    """加载干净 backbone（无 LoRA，用于 think 数据生成）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载 backbone: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map=device,
    )
    model.eval()
    print(f"  backbone 加载完成 ({device})")
    return model, tokenizer


def generate_think_response(
    model, tokenizer, messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    用干净 backbone 生成 think 回复（单条）。

    使用 Qwen3 原生模板（支持 think），不用 ensure_chat_template。
    原生模板的 generation prompt 包含 <think>\\n，所以生成内容从 think 内部开始。
    """
    results = generate_think_responses_batch(
        model, tokenizer, [messages], max_new_tokens, temperature, top_p,
    )
    return results[0]


def generate_think_responses_batch(
    model, tokenizer, messages_batch: list[list[dict]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """
    批量生成 think 回复。左侧 padding，一次推理多条。
    """
    # 构造 prompt texts
    prompt_texts = []
    for messages in messages_batch:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_texts.append(prompt_text)

    # 左侧 padding (decoder-only batch generation 需要)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    tokenizer.padding_side = original_side

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # 解码每条结果 (左侧 padding，所有 input 长度相同)
    input_len = inputs.input_ids.shape[1]
    results = []
    for i in range(len(messages_batch)):
        new_ids = outputs[i][input_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=False)

        if "<think>" not in response and "</think>" in response:
            response = "<think>\n" + response

        for tag in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            response = response.replace(tag, "")

        results.append(response.strip())

    return results


def validate_think_response(response: str) -> bool:
    """验证 think 回复质量"""
    if "<think>" not in response or "</think>" not in response:
        return False

    # tag 顺序正确
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    if think_start >= think_end:
        return False

    # think 块后必须有实际内容
    answer = response[think_end + len("</think>"):].strip()
    if len(answer) < 2:
        return False

    # 长度检查
    if len(response) < 10 or len(response) > 2000:
        return False

    return True


# ── 主流程 ──

def generate_think_episodes(
    rng: random.Random,
    num: int,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    max_retries: int = 3,
    batch_size: int = 16,
    # 类型比例 (fact / continuation / heartbeat / logic)
    ratio_fact: float = 0.55,
    ratio_continuation: float = 0.20,
    ratio_heartbeat: float = 0.15,
    ratio_logic: float = 0.10,
    # fact 子类型比例 (single / summary / reason / combination)
    ratio_fact_single: float = 0.40,
    ratio_fact_summary: float = 0.25,
    ratio_fact_reason: float = 0.20,
    ratio_fact_combination: float = 0.15,
    # 增量写入：传入文件则边生成边写，支持断点续生
    output_file: str = None,
) -> list[list[dict]]:
    """生成 think episodes (含 backbone 推理)，batch 推理加速，支持增量写入"""

    # 累积阈值
    TYPE_BOUNDS = [
        ratio_fact,
        ratio_fact + ratio_continuation,
        ratio_fact + ratio_continuation + ratio_heartbeat,
        1.0,
    ]
    FACT_SUBTYPES = ["single", "summary", "reason", "combination"]
    FACT_BOUNDS = [
        ratio_fact_single,
        ratio_fact_single + ratio_fact_summary,
        ratio_fact_single + ratio_fact_summary + ratio_fact_reason,
        1.0,
    ]

    def _build_one(rng):
        """构造单个 episode，返回 (turns, messages)"""
        r = rng.random()
        if r < TYPE_BOUNDS[0]:  # fact 推理
            sr = rng.random()
            subtype = FACT_SUBTYPES[0]
            for st, b in zip(FACT_SUBTYPES, FACT_BOUNDS):
                if sr < b:
                    subtype = st
                    break
            nf = rng.randint(2, 5) if subtype != "single" else rng.randint(1, 3)
            return build_fact_think_episode(rng, nf, subtype)
        elif r < TYPE_BOUNDS[1]:
            return build_continuation_episode(rng)
        elif r < TYPE_BOUNDS[2]:
            return build_heartbeat_episode(rng)
        else:
            return build_pure_logic_episode(rng)

    # 断点续生：检查已有数据
    existing = 0
    if output_file and Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        if existing >= num:
            print(f"  [断点续生] 已有 {existing}/{num} 条，跳过")
            return None
        print(f"  [断点续生] 已有 {existing}/{num} 条，继续生成剩余 {num - existing} 条")
        # 快进 rng 到正确位置
        for _ in range(existing):
            _build_one(rng)  # 消耗相同的 rng 序列

    episodes = []
    failed = 0
    fout = None
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fout = open(output_file, "a", encoding="utf-8")

    remaining = num - existing
    retry_queue = []
    pbar = tqdm(total=num, initial=existing, desc="生成 think episodes")

    try:
        pos = 0
        while pos < remaining:
            # 构造一个 batch 的 episodes
            cur_batch = min(batch_size, remaining - pos)
            batch_turns = []
            batch_messages = []
            for _ in range(cur_batch):
                turns, messages = _build_one(rng)
                batch_turns.append(turns)
                batch_messages.append(messages)

            # batch 推理
            try:
                responses = generate_think_responses_batch(
                    model, tokenizer, batch_messages, max_new_tokens,
                )
            except Exception as e:
                print(f"\n  Batch 推理失败: {e}，降级为逐条处理")
                responses = []
                for messages in batch_messages:
                    try:
                        r = generate_think_response(model, tokenizer, messages, max_new_tokens)
                        responses.append(r)
                    except Exception:
                        responses.append("")

            # 处理结果
            failed_in_batch = []
            for j in range(cur_batch):
                response = responses[j]
                if validate_think_response(response):
                    batch_turns[j][-1]["assistant"] = response
                    episodes.append(batch_turns[j])
                    if fout:
                        fout.write(episode_to_jsonl(batch_turns[j]) + "\n")
                        fout.flush()
                else:
                    failed += 1
                    failed_in_batch.append((batch_turns[j], batch_messages[j]))

            retry_queue.extend(failed_in_batch)
            pos += cur_batch
            pbar.update(cur_batch)

        # 补生成：逐条重试失败的 episodes
        if retry_queue:
            print(f"\n  补生成 {len(retry_queue)} 条失败 episodes...")
            recovered = 0
            for turns, messages in tqdm(retry_queue, desc="补生成"):
                try:
                    response = generate_think_response(
                        model, tokenizer, messages, max_new_tokens,
                    )
                    if validate_think_response(response):
                        turns[-1]["assistant"] = response
                        episodes.append(turns)
                        if fout:
                            fout.write(episode_to_jsonl(turns) + "\n")
                            fout.flush()
                        recovered += 1
                except Exception:
                    pass
            failed -= recovered
            print(f"  补回 {recovered} 条，最终失败 {failed} 条")

    finally:
        pbar.close()
        if fout:
            fout.close()

    if failed > 0:
        print(f"  {failed}/{num} episodes 最终失败 (已跳过)")

    return episodes


def generate_think_data(
    out_dir: str,
    num_think: int = 5000,
    num_memory: int = 5000,
    num_val_think: int = 100,
    num_val_memory: int = 100,
    model_path: str = "./models/qwen3-0.6b",
    device: str = "cuda",
    max_new_tokens: int = 512,
    seed: int = 42,
    memory_max_turns: int = 8,
    memory_num_facts: int = 5,
    memory_entity_ratio: float = 0.2,
    memory_recall_ratio: float = 0.2,
    memory_ai_recall_ratio: float = 0.5,
    memory_overwrite_ratio: float = 0.2,
    memory_same_category: float = 0.3,
    # think 类型比例
    ratio_fact: float = 0.55,
    ratio_continuation: float = 0.20,
    ratio_heartbeat: float = 0.15,
    ratio_logic: float = 0.10,
    gen_batch_size: int = 8,
):
    """生成混合训练数据 (memory + think)"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 加载 backbone
    model, tokenizer = load_backbone(model_path, device)

    # think 数据增量写入到临时文件，支持断点续生
    think_train_path = str(out_path / "_think_train.jsonl")
    think_val_path = str(out_path / "_think_val.jsonl")

    # 生成 think 数据
    print(f"\n=== 生成 Think 训练数据 ({num_think} episodes) ===")
    rng_train = random.Random(seed)
    think_ratio_kwargs = dict(
        ratio_fact=ratio_fact,
        ratio_continuation=ratio_continuation,
        ratio_heartbeat=ratio_heartbeat,
        ratio_logic=ratio_logic,
    )
    generate_think_episodes(
        rng_train, num_think, model, tokenizer, max_new_tokens,
        batch_size=gen_batch_size,
        **think_ratio_kwargs,
        output_file=think_train_path,
    )

    print(f"\n=== 生成 Think 验证数据 ({num_val_think} episodes) ===")
    rng_val = random.Random(seed + 10000)
    generate_think_episodes(
        rng_val, num_val_think, model, tokenizer, max_new_tokens,
        batch_size=gen_batch_size,
        **think_ratio_kwargs,
        output_file=think_val_path,
    )

    # 释放 backbone 显存
    del model
    torch.cuda.empty_cache()

    # 生成记忆数据 (复用 generate_memory_data.generate_data)
    print(f"\n=== 生成 Memory 数据 ({num_memory} + {num_val_memory}) ===")
    memory_dir = str(out_path / "_memory_tmp")
    memory_train_path, memory_val_path = generate_data(
        out_dir=memory_dir,
        num_train=num_memory,
        num_val=num_val_memory,
        num_facts=memory_num_facts,
        entity_ratio=memory_entity_ratio,
        recall_ratio=memory_recall_ratio,
        ai_recall_ratio=memory_ai_recall_ratio,
        overwrite_ratio=memory_overwrite_ratio,
        same_category=memory_same_category,
        min_distance=1,
        max_distance=10,
        max_turns=memory_max_turns,
        seed=seed + 20000,
    )

    # 混合并写出
    print(f"\n=== 混合输出 ===")
    for split, think_path, mem_path in [
        ("train", think_train_path, memory_train_path),
        ("val", think_val_path, memory_val_path),
    ]:
        with open(think_path, "r", encoding="utf-8") as f:
            think_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # 验证 JSON 完整性
                    think_lines.append(line)
                except json.JSONDecodeError:
                    pass  # 跳过被截断的不完整行

        with open(mem_path, "r", encoding="utf-8") as f:
            memory_lines = [line.strip() for line in f if line.strip()]

        # 混合打乱
        all_lines = memory_lines + think_lines
        rng_mix = random.Random(seed + (30000 if split == "train" else 40000))
        rng_mix.shuffle(all_lines)

        path = out_path / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")

        print(f"  {split}: {len(think_lines)} think + {len(memory_lines)} memory"
              f" = {len(all_lines)} episodes → {path}")

    # 清理临时文件
    shutil.rmtree(memory_dir, ignore_errors=True)
    for tmp in [think_train_path, think_val_path]:
        Path(tmp).unlink(missing_ok=True)

    return str(out_path / "train.jsonl"), str(out_path / "val.jsonl")


def preview_think_episode(turns: list[dict], idx: int, messages: list[dict]):
    """预览一个 think episode"""
    print(f"\n{'='*60}")
    print(f"  Think Episode {idx}")
    print(f"{'='*60}")
    print(f"  [训练数据]")
    for i, turn in enumerate(turns):
        loss = " [LOSS]" if turn.get("train_loss") else ""
        u = turn["user"][:80] + ("..." if len(turn["user"]) > 80 else "")
        a = turn["assistant"][:80] + ("..." if len(turn["assistant"]) > 80 else "")
        print(f"    Turn {i+1}{loss}")
        print(f"      User:      {u}")
        print(f"      Assistant: {a}")
    print(f"  [Backbone Messages]")
    for msg in messages:
        c = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
        print(f"    {msg['role']}: {c}")
    print()


def main():
    """
    直接运行只支持 --preview 预览。
    正式生成请用统一入口:
        python scripts/generate_data.py --config configs/curriculum_qwen.yaml --stage 14_think
    """
    parser = argparse.ArgumentParser(
        description="Think 数据预览 (正式生成请用 generate_data.py)")
    parser.add_argument("--preview", type=int, default=3,
                        help="预览 N 条 episode（不写文件，不推理）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    type_names = ["fact-single", "fact-summary", "fact-reason",
                  "continuation", "heartbeat", "logic"]
    builders = [
        lambda: build_fact_think_episode(rng, 3, "single"),
        lambda: build_fact_think_episode(rng, 4, "summary"),
        lambda: build_fact_think_episode(rng, 3, "reason"),
        lambda: build_continuation_episode(rng),
        lambda: build_heartbeat_episode(rng),
        lambda: build_pure_logic_episode(rng),
    ]
    for i in range(args.preview):
        idx = i % len(builders)
        turns, messages = builders[idx]()
        turns[-1]["assistant"] = "<think>（backbone 推理占位）</think>这是回答。"
        print(f"  类型: {type_names[idx]}")
        preview_think_episode(turns, i, messages)
        print(f"  JSONL 头 200 字符: {episode_to_jsonl(turns)[:200]}...")


if __name__ == "__main__":
    main()
