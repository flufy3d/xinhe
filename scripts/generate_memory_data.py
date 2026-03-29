"""
生成记忆训练数据 (JSONL)

输出格式 (ShareGPT 兼容):
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "train_loss": true/false}, ...]}

每条数据 = 一个多轮对话 episode:
  turn 1: 用户告诉信息 → assistant 确认 (train_loss=false)
  turn 2~N-1: 闲聊填充 (train_loss=false)
  turn N: 用户提问 → assistant 凭记忆回答 (train_loss=true)

关键设计: 每个 episode 的事实都是随机生成的，LoRA 无法记忆具体答案，
只能学会从 state 读取信息的通用技能。

用法:
    python scripts/generate_memory_data.py
    python scripts/generate_memory_data.py --num-train 5000 --num-val 500 --max-distance 6
    python scripts/generate_memory_data.py --preview 3   # 只预览，不写文件
"""
import argparse
import json
import random
from pathlib import Path


# ── 随机姓名素材 ──

SURNAMES = [
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈",
    "褚", "卫", "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许",
    "何", "吕", "施", "张", "孔", "曹", "严", "华", "金", "魏",
    "陶", "姜", "戚", "谢", "邹", "喻", "柏", "水", "窦", "章",
    "云", "苏", "潘", "葛", "奚", "范", "彭", "郎", "鲁", "韦",
    "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳",
    "酆", "鲍", "史", "唐", "费", "廉", "岑", "薛", "雷", "贺",
    "倪", "汤", "滕", "殷", "罗", "毕", "郝", "邬", "安", "常",
    "乐", "于", "时", "傅", "皮", "卞", "齐", "康", "伍", "余",
    "元", "卜", "顾", "孟", "平", "黄", "和", "穆", "萧", "尹",
    "姚", "邵", "湛", "汪", "祁", "毛", "禹", "狄", "米", "贝",
    "明", "臧", "计", "伏", "成", "戴", "谈", "宋", "茅", "庞",
    "熊", "纪", "舒", "屈", "项", "祝", "董", "梁", "杜", "阮",
    "蓝", "闵", "席", "季", "麻", "强", "贾", "路", "娄", "危",
    "江", "童", "颜", "郭", "梅", "盛", "林", "刁", "钟", "徐",
    "邱", "骆", "高", "夏", "蔡", "田", "樊", "胡", "凌", "霍",
    "虞", "万", "支", "柯", "昝", "管", "卢", "莫", "经", "房",
    "裘", "缪", "干", "解", "应", "宗", "丁", "宣", "贲", "邓",
    "郁", "单", "杭", "洪", "包", "诸", "左", "石", "崔", "吉",
    "钮", "龚", "程", "嵇", "邢", "滑", "裴", "陆", "荣", "翁",
]

GIVEN_CHARS = [
    "伟", "芳", "娜", "秀", "英", "敏", "静", "丽", "强", "磊",
    "洋", "勇", "艳", "杰", "娟", "涛", "明", "超", "兰", "霞",
    "平", "刚", "桂", "文", "辉", "玲", "华", "红", "军", "燕",
    "萍", "建", "春", "琴", "云", "飞", "峰", "凤", "林", "鑫",
    "波", "健", "彬", "斌", "宇", "浩", "然", "博", "宏", "志",
    "海", "岩", "鹏", "旭", "俊", "哲", "睿", "翔", "晨", "辰",
    "阳", "凯", "昊", "龙", "瑞", "雪", "梅", "莹", "倩", "颖",
    "琳", "璐", "薇", "婷", "欣", "悦", "妍", "佳", "雨", "思",
    "涵", "蕊", "馨", "怡", "诗", "梦", "宁", "晴", "瑶", "萌",
    "洁", "蓉", "露", "菲", "寒", "冰", "月", "星", "风", "晓",
    "天", "正", "德", "义", "礼", "智", "信", "仁", "勤", "和",
    "安", "泰", "康", "裕", "福", "祥", "荣", "昌", "盛", "兴",
    "国", "栋", "良", "成", "光", "达", "永", "长", "新", "胜",
    "学", "才", "松", "柏", "茂", "进", "舟", "帆", "恒", "毅",
    "豪", "远", "航", "程", "锦", "绣", "昕", "彤", "曦", "妮",
    "璇", "琪", "萱", "蓓", "蕾", "苒", "葵", "茜", "莲", "竹",
]

# ── 中国地级市 ──

CITIES_LARGE = [
    "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安",
    "南京", "重庆", "苏州", "长沙", "青岛", "厦门", "昆明", "大连",
    "天津", "沈阳", "哈尔滨", "长春", "济南", "郑州", "石家庄", "太原",
    "合肥", "福州", "南昌", "兰州", "贵阳", "南宁", "海口", "银川",
    "西宁", "呼和浩特", "乌鲁木齐", "拉萨", "无锡", "常州", "徐州",
    "扬州", "南通", "镇江", "泰州", "盐城", "淮安", "宿迁", "连云港",
    "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "台州", "丽水",
    "衢州", "舟山", "芜湖", "蚌埠", "淮南", "马鞍山", "铜陵", "安庆",
    "黄山", "滁州", "阜阳", "宿州", "亳州", "池州", "宣城", "六安",
    "烟台", "潍坊", "临沂", "淄博", "济宁", "泰安", "威海", "日照",
    "德州", "聊城", "滨州", "菏泽", "枣庄", "东营", "珠海", "佛山",
    "东莞", "中山", "惠州", "江门", "湛江", "茂名", "肇庆", "汕头",
    "揭阳", "梅州", "韶关", "清远", "河源", "阳江", "潮州", "云浮",
    "洛阳", "开封", "平顶山", "安阳", "鹤壁", "新乡", "焦作", "濮阳",
    "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口", "驻马店",
    "桂林", "柳州", "北海", "梧州", "钦州", "百色", "玉林", "贺州",
    "株洲", "湘潭", "衡阳", "邵阳", "岳阳", "常德", "张家界", "益阳",
    "郴州", "永州", "怀化", "娄底", "遵义", "六盘水", "安顺", "铜仁",
    "曲靖", "玉溪", "保山", "昭通", "丽江", "临沧", "大理", "德宏",
    "宜宾", "绵阳", "德阳", "南充", "乐山", "泸州", "达州", "遂宁",
    "内江", "自贡", "广元", "眉山", "攀枝花", "雅安", "巴中", "资阳",
    "咸阳", "宝鸡", "渭南", "延安", "汉中", "榆林", "安康", "商洛",
    "大同", "阳泉", "长治", "晋城", "朔州", "晋中", "运城", "忻州",
    "临汾", "吕梁", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口",
    "承德", "廊坊", "衡水", "沧州", "鞍山", "抚顺", "本溪", "丹东",
    "锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭", "朝阳", "葫芦岛",
    "吉林", "四平", "辽源", "通化", "白山", "松原", "白城", "延边",
    "齐齐哈尔", "牡丹江", "佳木斯", "大庆", "鸡西", "双鸭山", "伊春",
    "七台河", "鹤岗", "黑河", "绥化", "大兴安岭", "漳州", "泉州",
    "三明", "莆田", "南平", "龙岩", "宁德", "赣州", "吉安", "宜春",
    "抚州", "上饶", "景德镇", "萍乡", "新余", "鹰潭",
]


# ── 随机生成函数 ──

def random_name(rng: random.Random) -> str:
    """随机姓+1~2字名，组合空间 200×160×160 ≈ 500万"""
    surname = rng.choice(SURNAMES)
    given = rng.choice(GIVEN_CHARS)
    if rng.random() < 0.6:
        given += rng.choice(GIVEN_CHARS)
    return surname + given


def random_number(rng: random.Random) -> str:
    """随机 4~6 位数字"""
    length = rng.randint(4, 6)
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def random_city(rng: random.Random) -> str:
    return rng.choice(CITIES_LARGE)


# ── 陈述模板 ──

FACT_TEMPLATES = {
    "name": [
        ("我叫{v}。", "好的，{v}，很高兴认识你！"),
        ("你可以叫我{v}。", "好的，{v}，我记住了！"),
        ("我的名字是{v}。", "你好{v}！我记住你的名字了。"),
    ],
    "number": [
        ("我的编号是{v}。", "好的，我记住了，你的编号是{v}。"),
        ("请记住我的号码：{v}。", "收到，{v}，我记下了。"),
        ("我的识别码是{v}。", "好的，{v}，已经记住了。"),
    ],
    "city": [
        ("我住在{v}。", "好的，{v}是个好地方！"),
        ("我现在在{v}生活。", "在{v}生活一定很不错吧！"),
        ("我的家在{v}。", "{v}啊，我知道了！"),
    ],
}

# ── 回忆提问模板 ──

RECALL_TEMPLATES = {
    "name": [
        ("我叫什么名字？", "你叫{v}。"),
        ("你还记得我的名字吗？", "当然记得，你叫{v}。"),
        ("请问我叫什么？", "你叫{v}呀。"),
    ],
    "number": [
        ("我的编号是什么？", "你的编号是{v}。"),
        ("你记得我的号码吗？", "你的号码是{v}。"),
        ("我的识别码是多少？", "你的识别码是{v}。"),
    ],
    "city": [
        ("我住在哪里？", "你住在{v}。"),
        ("你记得我在哪个城市吗？", "你在{v}。"),
        ("我的家在哪？", "你的家在{v}。"),
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
    """随机生成 num_facts 个不同类别的事实（每次都是全新的随机值）。"""
    generators = {
        "name": lambda: random_name(rng),
        "number": lambda: random_number(rng),
        "city": lambda: random_city(rng),
    }
    categories = rng.sample(list(generators.keys()), min(num_facts, len(generators)))
    return [{"category": cat, "value": generators[cat]()} for cat in categories]


def make_turn(template_pair: tuple[str, str], value: str, train_loss: bool = False) -> dict:
    """用模板生成一轮 user+assistant 对话。"""
    user_text = template_pair[0].format(v=value)
    asst_text = template_pair[1].format(v=value)
    return {"user": user_text, "assistant": asst_text, "train_loss": train_loss}


def generate_episode(
    rng: random.Random,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    num_facts: int = 1,
) -> list[dict]:
    """
    生成一个 episode 的对话轮次。

    结构: [告知事实] [闲聊填充 x distance] [回忆提问(train_loss=true)] [闲聊补充...]
    """
    facts = sample_facts(rng, num_facts)
    turns = []

    # 告知阶段: 每个事实一轮 (计算 loss，让模型学会回复确认)
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        turn = make_turn(template, fact["value"], train_loss=True)
        turns.append(turn)

    # 闲聊填充 (不计算 loss)
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆阶段: 随机挑一个事实提问 (train_loss=True!)
    recall_fact = rng.choice(facts)
    cat = recall_fact["category"]
    template = rng.choice(RECALL_TEMPLATES[cat])
    turn = make_turn(template, recall_fact["value"], train_loss=True)
    turns.append(turn)

    # 补充闲聊到 max_turns (不计算 loss)
    while len(turns) < max_turns:
        filler = rng.choice(FILLERS)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    return turns[:max_turns]


def episode_to_jsonl(turns: list[dict]) -> str:
    """将轮次列表转为 JSONL 行 (ShareGPT 格式 + train_loss 标记)。"""
    conversations = []
    for turn in turns:
        conversations.append({"role": "user", "content": turn["user"]})
        conversations.append({
            "role": "assistant",
            "content": turn["assistant"],
            "train_loss": turn.get("train_loss", True),
        })
    return json.dumps({"conversations": conversations}, ensure_ascii=False)


def preview_episode(turns: list[dict], idx: int = 0):
    """打印一个 episode 的内容。"""
    print(f"\n{'='*60}")
    print(f"  Episode {idx}")
    print(f"{'='*60}")
    for i, turn in enumerate(turns):
        loss_marker = " [LOSS]" if turn.get("train_loss") else ""
        print(f"  [Turn {i+1}]{loss_marker}")
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

    out_dir = Path(args.out_dir)

    def gen_episodes(rng: random.Random, num: int):
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
        rng = random.Random(args.seed)
        episodes = gen_episodes(rng, args.preview)
        for i, ep in enumerate(episodes):
            preview_episode(ep, i)
        print(f"JSONL 示例:")
        print(episode_to_jsonl(episodes[0]))
        return

    # 生成并写入文件 (train/val 用不同种子，确保无重叠)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, num, seed in [("train", args.num_train, args.seed),
                              ("val", args.num_val, args.seed + 10000)]:
        rng = random.Random(seed)
        episodes = gen_episodes(rng, num)
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(episode_to_jsonl(ep) + "\n")
        print(f"  {split}: {num} episodes → {path}")

    print(f"\n完成! 数据格式 (ShareGPT 兼容):")
    print(f'  {{"conversations": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "...", "train_loss": true}}, ...]}}')
    print(f"\n关键设计: 只有 recall turn 标记 train_loss=true，其余 turn 不参与 loss 计算")
    print(f"未来加入真实数据: 只需追加同格式 JSONL 行到 train.jsonl / val.jsonl")


if __name__ == "__main__":
    main()
