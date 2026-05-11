"""生成 write-0/read-3 微型召回探针数据集。

每个 episode 4 turns,严格固定结构:
  turn 0 (write):    user 提一个 entity → assistant 复述并记下
  turn 1, 2 (filler): 中性寒暄,train_loss="lm_only" 仅低权 LM 信号
  turn 3 (read):     user 问那个 entity → assistant 简短答出 entity

读 turn 的 user_msg 不含 entity 字符串 → validate_memory.py 标 is_recall=True。
写 turn 的 user_msg 含 entity 字符串 → is_recall=False(基线信号)。

目的:在 tbptt_turns=max_turns=4(全 BPTT 无 detach)+ 小数据 + 固定 distance=3
的极简设置下,看架构能不能学到跨 turn 召回 — 拟合 = 架构活着但被 TBPTT/规模杀,
全 0 = LoRA + 16 fresh_mem 表征断层。
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path


# ── entity 池 ──

# 英文人名:Qwen tokenizer 通常 1-2 BPE,first-token argmax 信号干净
NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Liam", "Maya", "Noah", "Olga", "Paul",
    "Quinn", "Rita", "Sam", "Tom", "Uma", "Vera", "Will", "Xena",
    "Yan", "Zoe", "Aaron", "Beth", "Cliff", "Dana", "Ethan", "Fiona",
    "George", "Hannah", "Iris", "Jake", "Kira", "Leo", "Mia", "Nate",
]

# 中文短名(2-3 字):Qwen tokenizer 中文按字切,2 字名 = 2 tokens
NAMES_CN = [
    "李雷", "韩梅", "张华", "王芳", "刘洋", "陈晨", "杨光", "黄莺",
    "周林", "吴楠", "郑伟", "赵敏", "孙磊", "马超", "朱明", "胡静",
    "高远", "林峰", "何雨", "罗琳",
]

# 城市
CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京",
    "武汉", "重庆", "天津", "苏州", "厦门", "青岛", "大连", "长沙",
]

# 食物
FOODS = [
    "寿司", "披萨", "烧烤", "火锅", "炸鸡", "牛排", "拉面", "饺子",
    "煎饼", "馄饨", "汉堡", "沙拉", "粥", "面包",
]

# 品牌
BRANDS = [
    "Sony", "Apple", "Nike", "Adidas", "Samsung", "Tesla", "BMW", "Toyota",
    "Lenovo", "Huawei", "Xiaomi", "Canon", "Dell", "Asus",
]

# 颜色
COLORS = [
    "深红", "湖蓝", "墨绿", "金黄", "雪白", "炭灰", "玫红", "孔雀蓝",
    "祖母绿", "鹅黄", "珊瑚粉",
]


# ── 4 种 entity 类型 + 对应模板 ──
# 每个类型给定:(entity 池, write 模板, read 问句)
# write 模板必须使 user_msg 含 entity(→ is_recall=False, write 基线信号)
# read 问句必须不含 entity(→ is_recall=True, 真实跨 turn 召回)
ENTITY_PATTERNS = [
    {
        "type": "name_en",
        "pool": NAMES,
        "user_write": "我叫{entity}。",
        "asst_write": "好的,我记住了,你叫{entity}。",
        "user_read": "我叫什么名字?",
        "asst_read": "{entity}。",
    },
    {
        "type": "name_cn",
        "pool": NAMES_CN,
        "user_write": "我叫{entity}。",
        "asst_write": "嗯,你叫{entity},我记下了。",
        "user_read": "我叫什么名字?",
        "asst_read": "{entity}。",
    },
    {
        "type": "city",
        "pool": CITIES,
        "user_write": "我老家在{entity}。",
        "asst_write": "了解,你老家在{entity}。",
        "user_read": "我老家在哪里?",
        "asst_read": "{entity}。",
    },
    {
        "type": "food",
        "pool": FOODS,
        "user_write": "我最喜欢吃{entity}。",
        "asst_write": "好的,你最喜欢吃{entity},我记住了。",
        "user_read": "我最喜欢吃什么?",
        "asst_read": "{entity}。",
    },
    {
        "type": "brand",
        "pool": BRANDS,
        "user_write": "我平时用{entity}的产品。",
        "asst_write": "嗯,你用{entity}的产品,记下了。",
        "user_read": "我平时用什么牌子?",
        "asst_read": "{entity}。",
    },
    {
        "type": "color",
        "pool": COLORS,
        "user_write": "我喜欢的颜色是{entity}。",
        "asst_write": "好的,你喜欢{entity},记住了。",
        "user_read": "我喜欢什么颜色?",
        "asst_read": "{entity}。",
    },
]


# ── filler turns ──
# train_loss="lm_only" → lm_weight=0.1,仅给 backbone 微弱 LM 信号,
# 主要 loss 还是 turn 0/3 上的 entity 信号
FILLERS = [
    ("今天天气怎么样?", "今天天气不错,阳光明媚。"),
    ("能帮我推荐一本书吗?", "我推荐你看《三体》,科幻很经典。"),
    ("最近有什么新电影?", "最近上映了几部不错的科幻片。"),
    ("周末适合做什么运动?", "周末可以去跑步或者打球。"),
    ("怎么提高睡眠质量?", "睡前不要看手机,规律作息。"),
    ("怎么样能学好英语?", "多听多说多练,坚持很重要。"),
    ("有什么放松的方法?", "可以听音乐、散步,或者泡个澡。"),
    ("怎么做番茄炒蛋?", "鸡蛋打散炒熟,番茄切块翻炒,加盐调味。"),
    ("今天有什么计划?", "我想读一会儿书,然后整理一下房间。"),
    ("你觉得喝什么茶好?", "绿茶清淡,适合白天;红茶醇厚,适合午后。"),
    ("城里哪家咖啡店好?", "市中心几家精品咖啡都不错,可以试试。"),
    ("如何缓解压力?", "深呼吸、冥想、做喜欢的事情都有用。"),
    ("怎么管理时间?", "列清单,把重要的事情放前面做。"),
    ("有什么好的旅行地?", "国内推荐云南,国外可以考虑日本。"),
    ("怎么提高专注力?", "减少干扰、固定时段做一件事。"),
    ("健康饮食有什么原则?", "三餐规律,多吃蔬菜,少油少盐。"),
]


# ── 生成单 episode ──

def make_episode(seed: int) -> dict:
    rng = random.Random(seed)
    pat = rng.choice(ENTITY_PATTERNS)
    entity = rng.choice(pat["pool"])

    user_write_text = pat["user_write"].format(entity=entity)
    asst_write_text = pat["asst_write"].format(entity=entity)
    user_read_text = pat["user_read"]
    asst_read_text = pat["asst_read"].format(entity=entity)

    # 防御:user_read 不应含 entity(否则 strict eval 会把 read 错标成 write)
    assert entity not in user_read_text, \
        f"BUG: read user_msg 含 entity ({entity}) → 会被标 is_recall=False"

    # 在 assistant content 内定位 entity → value_span(char 坐标)
    w_s = asst_write_text.find(entity)
    w_e = w_s + len(entity)
    assert w_s >= 0, f"write entity 未找到: {entity} in {asst_write_text}"

    r_s = asst_read_text.find(entity)
    r_e = r_s + len(entity)
    assert r_s >= 0, f"read entity 未找到: {entity} in {asst_read_text}"

    # 两个 filler — 必须不同
    f1, f2 = rng.sample(FILLERS, 2)

    return {
        "sample_id": uuid.uuid4().hex[:12],
        "stage": "0",
        "skeleton_id": "PROBE",
        "meta": {
            "n_turns": 4,
            "target_turns": 4,
            "distance_bucket": "far",  # write@turn0 → read@turn3,固定距离 3
            "distance_buckets_detail": {"PROBE": {"bucket": "far", "turns": 3}},
            "entity_type": pat["type"],
            "entity": entity,
        },
        "conversations": [
            # turn 0:write
            {"role": "user", "content": user_write_text},
            {
                "role": "assistant",
                "content": asst_write_text,
                "train_loss": "true",
                "value": [entity],
                "value_span": [[w_s, w_e]],
                "value_tier": "hard",
                "weight_per_span": 1.0,
            },
            # turn 1:filler
            {"role": "user", "content": f1[0]},
            {
                "role": "assistant",
                "content": f1[1],
                "train_loss": "lm_only",
                "value": None,
                "value_span": [],
                "value_tier": None,
                "weight_per_span": 0.0,
            },
            # turn 2:filler
            {"role": "user", "content": f2[0]},
            {
                "role": "assistant",
                "content": f2[1],
                "train_loss": "lm_only",
                "value": None,
                "value_span": [],
                "value_tier": None,
                "weight_per_span": 0.0,
            },
            # turn 3:read
            {"role": "user", "content": user_read_text},
            {
                "role": "assistant",
                "content": asst_read_text,
                "train_loss": "true",
                "value": [entity],
                "value_span": [[r_s, r_e]],
                "value_tier": "hard",
                "weight_per_span": 1.0,
            },
        ],
    }


# ── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/recall_probe")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    rng = random.Random(args.seed)

    # train
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(args.n_train):
            ep = make_episode(rng.randint(0, 2**31))
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"  写 {args.n_train} train episodes → {train_path}")

    # val(用错开 seed 段)
    rng_val = random.Random(args.seed + 10_000_000)
    with open(val_path, "w", encoding="utf-8") as f:
        for i in range(args.n_val):
            ep = make_episode(rng_val.randint(0, 2**31))
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"  写 {args.n_val} val episodes → {val_path}")

    # 样本检查:打第一条
    with open(train_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n  样本预览(train[0]):")
    print(f"    entity={first['meta']['entity']} type={first['meta']['entity_type']}")
    for t in first["conversations"]:
        role = t["role"]
        content = t["content"]
        loss = t.get("train_loss", "—")
        print(f"    [{role:9s}] [{loss:8s}] {content}")


if __name__ == "__main__":
    main()
