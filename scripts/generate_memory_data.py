"""
生成记忆训练数据 (JSONL)

输出格式 (ShareGPT 兼容):
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

每条数据 = 一个多轮对话 episode:
  turn 1: 用户告诉信息 → assistant 确认
  turn 2~N-1: 闲聊填充
  turn N: 用户提问 → assistant 凭记忆回答

用法:
    python scripts/generate_memory_data.py
    python scripts/generate_memory_data.py --num-train 5000 --num-val 500 --max-distance 6
    python scripts/generate_memory_data.py --preview 3   # 只预览，不写文件
"""
import argparse
import json
import random
from pathlib import Path


# ── 记忆素材库 ──

NAMES = ["小明", "小红", "张三", "李四", "王五", "小刚", "小芳", "小华",
         "阿杰", "小美", "大伟", "思思", "浩然", "雨萱", "子轩", "梓涵"]

CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安",
          "南京", "重庆", "苏州", "长沙", "青岛", "厦门", "昆明", "大连"]

FOODS = ["火锅", "烤鸭", "拉面", "寿司", "披萨", "饺子", "米粉", "煎饼",
         "烧烤", "麻辣烫", "炒饭", "螺蛳粉", "小笼包", "酸菜鱼", "糖醋排骨", "红烧肉"]

COLORS = ["红色", "蓝色", "绿色", "紫色", "黄色", "白色", "黑色", "橙色"]

HOBBIES = ["跑步", "读书", "画画", "弹吉他", "摄影", "游泳", "打篮球", "下棋",
           "看电影", "写代码", "弹钢琴", "爬山", "骑车", "做饭", "种花", "打游戏"]

PETS = ["小白", "旺财", "咪咪", "球球", "豆豆", "花花", "大黄", "小黑"]

PET_TYPES = ["猫", "狗", "仓鼠", "兔子"]

JOBS = ["程序员", "老师", "医生", "设计师", "工程师", "记者", "厨师", "律师"]

# ── 陈述模板 ──

FACT_TEMPLATES = {
    "name": [
        ("我叫{v}。", "好的，{v}，很高兴认识你！"),
        ("你可以叫我{v}。", "好的，{v}，我记住了！"),
        ("我的名字是{v}。", "你好{v}！我记住你的名字了。"),
    ],
    "city": [
        ("我住在{v}。", "好的，{v}是个好地方！"),
        ("我现在在{v}生活。", "在{v}生活一定很不错吧！"),
        ("我的家在{v}。", "{v}啊，我知道了！"),
    ],
    "food": [
        ("我最喜欢吃{v}。", "哇，{v}确实很好吃！"),
        ("{v}是我最爱的食物。", "{v}很受欢迎呢！"),
        ("说到吃的，我最爱{v}。", "{v}是个好选择！"),
    ],
    "color": [
        ("我最喜欢的颜色是{v}。", "{v}是个很好看的颜色！"),
        ("{v}是我最爱的颜色。", "我记住了，你喜欢{v}。"),
    ],
    "hobby": [
        ("我平时喜欢{v}。", "{v}是个很好的爱好！"),
        ("我的爱好是{v}。", "喜欢{v}的人都很有生活品味！"),
    ],
    "pet": [
        ("我养了一只{pt}，叫{v}。", "哇，{v}这个名字真可爱！"),
        ("我家有只{pt}叫{v}。", "{v}一定很可爱吧！"),
    ],
    "job": [
        ("我是一名{v}。", "做{v}一定很有意思！"),
        ("我的职业是{v}。", "{v}是个很不错的职业！"),
    ],
}

# ── 回忆提问模板 ──

RECALL_TEMPLATES = {
    "name": [
        ("我叫什么名字？", "你叫{v}。"),
        ("你还记得我的名字吗？", "当然记得，你叫{v}。"),
        ("请问我叫什么？", "你叫{v}呀。"),
    ],
    "city": [
        ("我住在哪里？", "你住在{v}。"),
        ("你记得我在哪个城市吗？", "你在{v}。"),
        ("我的家在哪？", "你的家在{v}。"),
    ],
    "food": [
        ("我喜欢吃什么？", "你最喜欢吃{v}。"),
        ("你记得我爱吃什么吗？", "你爱吃{v}。"),
        ("我最爱的食物是什么？", "你最爱{v}。"),
    ],
    "color": [
        ("我最喜欢什么颜色？", "你最喜欢{v}。"),
        ("你记得我喜欢的颜色吗？", "你喜欢{v}。"),
    ],
    "hobby": [
        ("我有什么爱好？", "你喜欢{v}。"),
        ("你还记得我平时喜欢做什么吗？", "你平时喜欢{v}。"),
    ],
    "pet": [
        ("我的宠物叫什么名字？", "你的{pt}叫{v}。"),
        ("你记得我家宠物的名字吗？", "你家{pt}叫{v}。"),
    ],
    "job": [
        ("我是做什么工作的？", "你是一名{v}。"),
        ("你记得我的职业吗？", "你是{v}。"),
    ],
}

# ── 闲聊填充 ──

FILLERS = [
    ("今天天气怎么样？", "今天天气不错，阳光明媚，适合出门走走。"),
    ("给我讲个笑话吧。", "好的！为什么程序员不喜欢户外？因为有太多bug。"),
    ("你觉得AI会取代人类吗？", "AI是工具，会帮助人类而非取代。每种技术都有其适用场景。"),
    ("推荐一部电影吧。", "推荐《星际穿越》，讲述了一段跨越时空的感人故事。"),
    ("你有什么建议给我？", "建议多读书多运动，保持好奇心，享受学习的过程。"),
    ("最近有什么新闻？", "科技领域一直在快速发展，AI和新能源是近年来的热门话题。"),
    ("给我推荐一首歌。", "推荐《晴天》，周杰伦的经典歌曲，旋律很好听。"),
    ("你觉得学什么技能好？", "编程是一项很实用的技能，能帮你解决很多问题。"),
    ("帮我想个周末计划。", "可以去公园散步，或者找一家新餐厅尝尝鲜，再看场电影。"),
    ("你对读书有什么看法？", "读书能拓宽视野，是最经济实惠的自我提升方式。"),
    ("说说你对运动的看法。", "运动有益身心健康，建议每天至少活动半小时。"),
    ("有什么好的学习方法？", "间隔重复和主动回忆是很有效的学习方法。"),
    ("你觉得睡眠重要吗？", "非常重要！充足的睡眠对记忆力和创造力都有很大帮助。"),
    ("今天吃什么好呢？", "可以试试做个家常菜，简单又健康。"),
    ("最近压力好大。", "适当休息很重要，可以试试深呼吸或者散步来放松。"),
    ("你喜欢什么季节？", "每个季节都有独特的美，春天万物复苏，秋天层林尽染。"),
]


def sample_facts(rng: random.Random, num_facts: int = 1) -> list[dict]:
    """随机抽取 num_facts 个不同类别的事实。"""
    pool = {
        "name": (NAMES, None),
        "city": (CITIES, None),
        "food": (FOODS, None),
        "color": (COLORS, None),
        "hobby": (HOBBIES, None),
        "pet": (PETS, PET_TYPES),
        "job": (JOBS, None),
    }
    categories = rng.sample(list(pool.keys()), min(num_facts, len(pool)))
    facts = []
    for cat in categories:
        values, extras = pool[cat]
        v = rng.choice(values)
        extra = {"pt": rng.choice(extras)} if extras else {}
        facts.append({"category": cat, "value": v, **extra})
    return facts


def make_turn(template_pair: tuple[str, str], value: str, **kwargs) -> dict:
    """用模板生成一轮 user+assistant 对话。"""
    user_text = template_pair[0].format(v=value, **kwargs)
    asst_text = template_pair[1].format(v=value, **kwargs)
    return {"user": user_text, "assistant": asst_text}


def generate_episode(
    rng: random.Random,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    num_facts: int = 1,
) -> list[dict]:
    """
    生成一个 episode 的对话轮次。

    结构: [告知事实] [闲聊填充 x distance] [回忆提问] [闲聊补充...]
    """
    facts = sample_facts(rng, num_facts)
    turns = []

    # 告知阶段: 每个事实一轮
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        extra = {"pt": fact.get("pt", "")}
        turn = make_turn(template, fact["value"], **extra)
        turns.append(turn)

    # 闲聊填充
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1]})

    # 回忆阶段: 随机挑一个事实提问
    recall_fact = rng.choice(facts)
    cat = recall_fact["category"]
    template = rng.choice(RECALL_TEMPLATES[cat])
    extra = {"pt": recall_fact.get("pt", "")}
    turn = make_turn(template, recall_fact["value"], **extra)
    turns.append(turn)

    # 补充闲聊到 max_turns
    while len(turns) < max_turns:
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1]})

    return turns[:max_turns]


def episode_to_jsonl(turns: list[dict]) -> str:
    """将轮次列表转为 JSONL 行 (ShareGPT 格式)。"""
    conversations = []
    for turn in turns:
        conversations.append({"role": "user", "content": turn["user"]})
        conversations.append({"role": "assistant", "content": turn["assistant"]})
    return json.dumps({"conversations": conversations}, ensure_ascii=False)


def preview_episode(turns: list[dict], idx: int = 0):
    """打印一个 episode 的内容。"""
    print(f"\n{'='*60}")
    print(f"  Episode {idx}")
    print(f"{'='*60}")
    for i, turn in enumerate(turns):
        print(f"  [Turn {i+1}]")
        print(f"    User:      {turn['user']}")
        print(f"    Assistant: {turn['assistant']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="生成记忆训练数据")
    parser.add_argument("--num-train", type=int, default=2000, help="训练集 episode 数量")
    parser.add_argument("--num-val", type=int, default=200, help="验证集 episode 数量")
    parser.add_argument("--max-turns", type=int, default=16, help="每个 episode 最大轮数")
    parser.add_argument("--min-distance", type=int, default=1, help="记忆→回忆最小间隔轮数")
    parser.add_argument("--max-distance", type=int, default=4, help="记忆→回忆最大间隔轮数")
    parser.add_argument("--num-facts", type=int, default=1, help="每个 episode 的事实数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data", help="输出目录")
    parser.add_argument("--preview", type=int, default=0, help="预览 N 条数据（不写文件）")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    def gen_episodes(num: int):
        episodes = []
        for _ in range(num):
            turns = generate_episode(
                rng,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                max_turns=args.max_turns,
                num_facts=args.num_facts,
            )
            episodes.append(turns)
        return episodes

    # 预览模式
    if args.preview > 0:
        episodes = gen_episodes(args.preview)
        for i, ep in enumerate(episodes):
            preview_episode(ep, i)
        print(f"JSONL 示例:")
        print(episode_to_jsonl(episodes[0]))
        return

    # 生成并写入文件
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, num in [("train", args.num_train), ("val", args.num_val)]:
        episodes = gen_episodes(num)
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(episode_to_jsonl(ep) + "\n")
        print(f"  {split}: {num} episodes → {path}")

    print(f"\n完成! 数据格式 (ShareGPT 兼容):")
    print(f'  {{"conversations": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]}}')
    print(f"\n未来加入真实数据: 只需追加同格式 JSONL 行到 train.jsonl / val.jsonl")


if __name__ == "__main__":
    main()
