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

from xinhe.data.think_lang import THINK_LANG, fact_summary, wrap_think


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


# ── 复姓 ──

COMPOUND_SURNAMES = [
    "欧阳", "司马", "上官", "诸葛", "东方", "公孙", "慕容", "皇甫",
    "令狐", "独孤", "南宫", "西门", "百里", "呼延", "端木", "轩辕",
    "长孙", "宇文", "尉迟", "澹台", "夏侯", "万俟", "司徒", "太史",
]

# ── 英文名 + 昵称 (堵字符集捷径) ──

ENGLISH_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Ruby", "Sam", "Tom", "Uma", "Vera", "Will", "Xena",
    "Yuki", "Zoe", "Alex", "Luna", "Max", "Lily", "Oscar", "Ella",
    "Ryan", "Chloe", "Ethan", "Sophia", "Liam", "Amy", "Kevin", "Jenny",
]

NICKNAMES = [
    "小飞侠", "大白", "阿呆", "小丸子", "豆豆", "糖糖", "果果", "团团",
    "圆圆", "皮皮", "乐乐", "欢欢", "萌萌", "贝贝", "蛋蛋", "球球",
    "A君", "小K", "Mr.Z", "Dr.X", "666", "007", "101", "Lucky",
]

# ── 随机生成函数 ──

def random_name(rng: random.Random) -> str:
    """随机姓名，1~4字中文 / 英文名 / 昵称，堵长度和字符集捷径"""
    r = rng.random()
    if r < 0.05:
        # 5%: 单字 (1字) — "夏", "龙"
        return rng.choice(GIVEN_CHARS)
    elif r < 0.15:
        # 10%: 英文名 / 昵称
        return rng.choice(ENGLISH_NAMES) if rng.random() < 0.6 else rng.choice(NICKNAMES)
    elif r < 0.30:
        # 15%: 复姓+双字名 (4字) — "欧阳明月"
        return rng.choice(COMPOUND_SURNAMES) + rng.choice(GIVEN_CHARS) + rng.choice(GIVEN_CHARS)
    elif r < 0.55:
        # 25%: 单姓+单字名 (2字) — "李明"
        return rng.choice(SURNAMES) + rng.choice(GIVEN_CHARS)
    else:
        # 45%: 单姓+双字名 (3字) — "李明月"
        return rng.choice(SURNAMES) + rng.choice(GIVEN_CHARS) + rng.choice(GIVEN_CHARS)


def random_number(rng: random.Random) -> str:
    """随机 1~8 位数字，长度多样化"""
    length = rng.randint(1, 8)
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def random_city(rng: random.Random) -> str:
    return rng.choice(CITIES_LARGE)


# ── 新增类别素材 (组合生成扩大池子) ──

CLASSIC_FOODS = [
    "火锅", "烤鸭", "麻辣烫", "饺子", "拉面", "小龙虾", "酸菜鱼",
    "宫保鸡丁", "麻婆豆腐", "鱼香肉丝", "煎饼", "包子", "馄饨",
    "凉皮", "肉夹馍", "臭豆腐", "螺蛳粉", "热干面", "刀削面",
    "三明治", "牛排", "意大利面", "咖喱饭", "炸鸡", "寿司", "披萨",
]

COOK_METHODS = [
    "红烧", "清炒", "水煮", "糖醋", "麻辣", "蒜蓉", "葱爆", "干煸",
    "酱焖", "香煎", "清蒸", "烤", "卤", "凉拌", "油炸", "炖",
]

INGREDIENTS = [
    "牛肉", "鸡肉", "排骨", "豆腐", "土豆", "茄子", "白菜", "鲈鱼",
    "虾", "猪肉", "羊肉", "鸡蛋", "冬瓜", "南瓜", "萝卜", "蘑菇",
    "木耳", "藕", "芹菜", "西兰花",
]

JOB_PREFIXES = ["资深", "实习", "高级", "初级", "首席", "助理", "全职", "兼职"]

JOBS = [
    "程序员", "教师", "医生", "律师", "设计师", "工程师", "记者", "厨师",
    "警察", "消防员", "护士", "会计", "司机", "建筑师", "摄影师", "作家",
    "画家", "音乐家", "导演", "演员", "销售", "翻译", "研究员", "飞行员",
    "快递员", "外卖员", "理发师", "园丁", "农民", "渔民", "木匠", "电工",
]

HOBBY_MODS = ["经常", "偶尔", "每天", "周末", "晚上", "一个人", "和朋友", "在家"]

HOBBIES = [
    "打篮球", "踢足球", "游泳", "跑步", "爬山", "骑自行车", "打羽毛球",
    "打乒乓球", "下棋", "钓鱼", "画画", "弹吉他", "弹钢琴", "唱歌",
    "看电影", "读书", "写作", "摄影", "旅行", "做饭", "养花", "打游戏",
    "滑雪", "冲浪", "瑜伽", "跳舞", "书法", "编程", "种菜", "看动漫",
]

PET_COLORS = [
    "白色的", "黑色的", "灰色的", "棕色的", "花色的",
    "橘色的", "奶油色的", "金色的", "黑白的", "三花",
]

PET_ANIMALS = [
    "猫", "狗", "兔子", "仓鼠", "鹦鹉", "乌龟", "金鱼", "柯基",
    "泰迪", "柴犬", "英短", "布偶猫", "边牧", "拉布拉多", "哈士奇",
    "比熊", "蓝猫", "暹罗猫", "龙猫", "刺猬",
]


def random_food(rng: random.Random) -> str:
    """做法×食材组合 (320种) + 经典菜名，堵小池记忆"""
    if rng.random() < 0.15:
        return rng.choice(CLASSIC_FOODS)
    return rng.choice(COOK_METHODS) + rng.choice(INGREDIENTS)

def random_job(rng: random.Random) -> str:
    """前缀×职业组合 (240种)，堵小池记忆"""
    if rng.random() < 0.3:
        return rng.choice(JOBS)
    return rng.choice(JOB_PREFIXES) + rng.choice(JOBS)

def random_hobby(rng: random.Random) -> str:
    """修饰×活动组合 (200种)，堵小池记忆"""
    if rng.random() < 0.3:
        return rng.choice(HOBBIES)
    return rng.choice(HOBBY_MODS) + rng.choice(HOBBIES)

def random_age(rng: random.Random) -> str:
    return str(rng.randint(1, 99))

def random_pet(rng: random.Random) -> str:
    """颜色×动物组合 (200种)，堵小池记忆"""
    if rng.random() < 0.2:
        return rng.choice(PET_ANIMALS)
    return rng.choice(PET_COLORS) + rng.choice(PET_ANIMALS)


# ── 陈述模板 ──

FACT_TEMPLATES = {
    "name": [
        ("我叫{v}。", "好的，{v}，很高兴认识你！"),
        ("你可以叫我{v}。", "好的，{v}，我记住了！"),
        ("我的名字是{v}。", "你好{v}！我记住你的名字了。"),
        ("大家都叫我{v}。", "好的，{v}，我也这么叫你！"),
        ("我是{v}。", "你好，{v}！"),
        ("称呼我{v}就好。", "好的，{v}！"),
        ("我姓名是{v}。", "好的，{v}，记住了！"),
        ("本人{v}。", "你好{v}，认识你很高兴！"),
        ("我名叫{v}。", "{v}，你好呀！"),
        ("叫我{v}吧。", "好的，{v}！"),
    ],
    "number": [
        ("我的编号是{v}。", "好的，我记住了，你的编号是{v}。"),
        ("请记住我的号码：{v}。", "收到，{v}，我记下了。"),
        ("我的识别码是{v}。", "好的，{v}，已经记住了。"),
        ("我的工号是{v}。", "工号{v}，记下了。"),
        ("编号{v}，帮我记着。", "好的，{v}已记录。"),
        ("我的号码是{v}。", "好的，号码{v}，已记住。"),
        ("记一下，{v}。", "好的，{v}，记住了。"),
        ("我的代号是{v}。", "代号{v}，收到。"),
        ("我是{v}号。", "好的，{v}号！"),
        ("我的学号是{v}。", "学号{v}，记住了。"),
    ],
    "city": [
        ("我住在{v}。", "好的，{v}是个好地方！"),
        ("我现在在{v}生活。", "在{v}生活一定很不错吧！"),
        ("我的家在{v}。", "{v}啊，我知道了！"),
        ("我家在{v}。", "好的，{v}，记住了！"),
        ("我老家是{v}的。", "{v}啊，好地方！"),
        ("我目前定居在{v}。", "在{v}定居不错！"),
        ("我是{v}人。", "好的，{v}人！"),
        ("我来自{v}。", "{v}来的啊，欢迎！"),
        ("我在{v}工作生活。", "好的，{v}，记住了。"),
        ("坐标{v}。", "好的，{v}！"),
    ],
    "food": [
        ("我喜欢吃{v}。", "好的，{v}很好吃！"),
        ("我最爱吃{v}。", "{v}确实美味！"),
        ("我最喜欢的食物是{v}。", "{v}，好选择！"),
        ("{v}是我的最爱。", "好的，你最爱{v}！"),
        ("我特别爱吃{v}。", "{v}，记住了！"),
        ("我平时最喜欢吃{v}。", "好的，{v}确实不错。"),
        ("说到吃的，我最喜欢{v}。", "{v}，好口味！"),
        ("我的口味偏好是{v}。", "好的，你喜欢{v}。"),
        ("没什么比{v}更好吃了。", "好的，你最爱{v}！"),
        ("要说最爱的美食，那必须是{v}。", "{v}，记住了！"),
    ],
    "job": [
        ("我是{v}。", "好的，{v}这个职业不错！"),
        ("我的职业是{v}。", "好的，你是{v}，记住了。"),
        ("我做{v}的。", "好的，{v}！"),
        ("我目前从事{v}工作。", "{v}工作，了解了。"),
        ("我的工作是{v}。", "好的，{v}，记住了。"),
        ("我是一名{v}。", "好的，你是{v}！"),
        ("职业是{v}。", "好的，{v}。"),
        ("我干{v}这行的。", "好的，{v}这行！"),
        ("我在做{v}。", "好的，{v}！"),
        ("我当{v}的。", "好的，你是{v}！"),
    ],
    "hobby": [
        ("我喜欢{v}。", "好的，{v}是个好爱好！"),
        ("我的爱好是{v}。", "{v}，不错的爱好！"),
        ("我平时喜欢{v}。", "好的，你喜欢{v}。"),
        ("我没事的时候喜欢{v}。", "{v}，很好的消遣！"),
        ("我最大的爱好就是{v}。", "好的，你最爱{v}！"),
        ("{v}是我的兴趣。", "好的，你对{v}感兴趣。"),
        ("我业余时间都在{v}。", "好的，{v}，记住了。"),
        ("我特别喜欢{v}。", "{v}，好爱好！"),
        ("闲暇时我一般{v}。", "好的，你喜欢{v}。"),
        ("我热爱{v}。", "好的，你热爱{v}！"),
    ],
    "age": [
        ("我今年{v}岁。", "好的，你{v}岁了。"),
        ("我{v}岁了。", "好的，{v}岁，记住了。"),
        ("我的年龄是{v}岁。", "好的，你{v}岁。"),
        ("本人{v}岁。", "好的，{v}岁！"),
        ("我已经{v}岁了。", "好的，你{v}岁了。"),
        ("年龄{v}。", "好的，{v}岁。"),
        ("我今年{v}了。", "好的，你今年{v}了。"),
        ("我快{v}岁了。", "好的，你快{v}岁了。"),
        ("我刚满{v}岁。", "好的，刚满{v}岁。"),
        ("我属于{v}岁这个年纪。", "好的，{v}岁。"),
    ],
    "pet": [
        ("我养了一只{v}。", "好的，{v}一定很可爱！"),
        ("我有一只{v}。", "好的，你养了{v}！"),
        ("我的宠物是{v}。", "好的，{v}，真不错！"),
        ("我家有只{v}。", "好的，{v}！"),
        ("我养了{v}当宠物。", "好的，{v}是好宠物！"),
        ("我在养{v}。", "好的，养{v}很有趣。"),
        ("我有个小{v}。", "好的，小{v}！"),
        ("家里养着一只{v}。", "好的，{v}！"),
        ("我是{v}的主人。", "好的，你有{v}！"),
        ("我的伙伴是一只{v}。", "好的，{v}是好伙伴！"),
    ],
}

# ── 覆写模板 ──

OVERWRITE_TEMPLATES = {
    "name": [
        ("不对，我其实叫{v}。", "好的，我记住了，你叫{v}。"),
        ("我改名了，现在叫{v}。", "好的，{v}，我更新了。"),
        ("叫我{v}吧。", "好的，{v}！"),
    ],
    "number": [
        ("我的编号改了，现在是{v}。", "好的，新编号{v}，已更新。"),
        ("不对，我的号码是{v}。", "收到，已更新为{v}。"),
    ],
    "city": [
        ("我搬家了，现在住在{v}。", "好的，{v}是个好地方！"),
        ("不对，我现在在{v}。", "好的，已更新为{v}。"),
    ],
    "food": [
        ("不对，我其实最爱吃{v}。", "好的，已更新为{v}。"),
        ("我改口味了，现在喜欢{v}。", "好的，你现在喜欢{v}。"),
    ],
    "job": [
        ("我换工作了，现在是{v}。", "好的，你现在是{v}了。"),
        ("不对，我现在做{v}。", "好的，已更新为{v}。"),
    ],
    "hobby": [
        ("我最近改了爱好，现在喜欢{v}。", "好的，你现在喜欢{v}。"),
        ("不对，我现在更喜欢{v}。", "好的，已更新为{v}。"),
    ],
    "age": [
        ("不对，我其实{v}岁。", "好的，你{v}岁，已更新。"),
        ("我记错了，我是{v}岁。", "好的，{v}岁，已更新。"),
    ],
    "pet": [
        ("我换宠物了，现在养{v}。", "好的，你现在养{v}了。"),
        ("不对，我现在养的是{v}。", "好的，已更新为{v}。"),
    ],
}

# ── 回忆提问模板 ──

RECALL_TEMPLATES = {
    "name": [
        ("我叫什么名字？", "你叫{v}。"),
        ("你还记得我的名字吗？", "当然记得，你叫{v}。"),
        ("请问我叫什么？", "你叫{v}呀。"),
        ("我是谁？", "你是{v}。"),
        ("你知道我叫什么吗？", "你叫{v}。"),
    ],
    "number": [
        ("我的编号是什么？", "你的编号是{v}。"),
        ("你记得我的号码吗？", "你的号码是{v}。"),
        ("我的识别码是多少？", "你的识别码是{v}。"),
        ("我是几号？", "你是{v}号。"),
        ("我的号码是什么？", "你的号码是{v}。"),
    ],
    "city": [
        ("我住在哪里？", "你住在{v}。"),
        ("你记得我在哪个城市吗？", "你在{v}。"),
        ("我的家在哪？", "你的家在{v}。"),
        ("我是哪里人？", "你是{v}人。"),
        ("我在哪生活？", "你在{v}生活。"),
    ],
    "food": [
        ("我喜欢吃什么？", "你喜欢吃{v}。"),
        ("你记得我爱吃什么吗？", "你爱吃{v}。"),
        ("我最爱的食物是？", "你最爱{v}。"),
        ("我喜欢什么美食？", "你喜欢{v}。"),
        ("我爱吃啥？", "你爱吃{v}。"),
    ],
    "job": [
        ("我的职业是什么？", "你的职业是{v}。"),
        ("我是做什么工作的？", "你是{v}。"),
        ("你记得我的工作吗？", "你是{v}。"),
        ("我是干什么的？", "你是{v}。"),
        ("我从事什么职业？", "你从事{v}。"),
    ],
    "hobby": [
        ("我的爱好是什么？", "你的爱好是{v}。"),
        ("你记得我喜欢做什么吗？", "你喜欢{v}。"),
        ("我平时喜欢干什么？", "你喜欢{v}。"),
        ("我有什么兴趣爱好？", "你喜欢{v}。"),
        ("我闲暇时做什么？", "你喜欢{v}。"),
    ],
    "age": [
        ("我多大了？", "你{v}岁了。"),
        ("你记得我的年龄吗？", "你{v}岁。"),
        ("我几岁了？", "你{v}岁了。"),
        ("我今年多大？", "你今年{v}岁。"),
        ("我的年龄是多少？", "你{v}岁。"),
    ],
    "pet": [
        ("我养了什么宠物？", "你养了{v}。"),
        ("你记得我的宠物吗？", "你的宠物是{v}。"),
        ("我有什么小动物？", "你有{v}。"),
        ("我家养了什么？", "你家养了{v}。"),
        ("我的宠物是什么？", "你的宠物是{v}。"),
    ],
}

# ── 实体定义 (我/你/他/她/它) ──

ENTITIES = [
    {"tell": "我", "recall_a": "你"},
    {"tell": "你", "recall_a": "我"},
    {"tell": "他", "recall_a": "他"},
    {"tell": "她", "recall_a": "她"},
    {"tell": "它", "recall_a": "它"},
]

# 实体模板：{e}=告知主语, {ea}=回答主语, {v}=值
ENTITY_FACT_TEMPLATES = {
    "name": [
        ("{e}叫{v}。", "好的，{ea}叫{v}！"),
        ("{e}的名字是{v}。", "好的，{ea}叫{v}，记住了。"),
        ("{e}名叫{v}。", "好的，{ea}叫{v}！"),
    ],
    "number": [
        ("{e}的编号是{v}。", "好的，{ea}的编号是{v}。"),
        ("{e}的号码是{v}。", "好的，{ea}是{v}号。"),
    ],
    "city": [
        ("{e}住在{v}。", "好的，{ea}住在{v}。"),
        ("{e}在{v}生活。", "好的，{ea}在{v}。"),
        ("{e}是{v}人。", "好的，{ea}是{v}人。"),
    ],
    "food": [
        ("{e}喜欢吃{v}。", "好的，{ea}喜欢吃{v}。"),
        ("{e}最爱吃{v}。", "好的，{ea}最爱{v}。"),
    ],
    "job": [
        ("{e}是{v}。", "好的，{ea}是{v}。"),
        ("{e}的职业是{v}。", "好的，{ea}是{v}。"),
    ],
    "hobby": [
        ("{e}喜欢{v}。", "好的，{ea}喜欢{v}。"),
        ("{e}的爱好是{v}。", "好的，{ea}喜欢{v}。"),
    ],
    "age": [
        ("{e}今年{v}岁。", "好的，{ea}{v}岁。"),
        ("{e}{v}岁了。", "好的，{ea}{v}岁。"),
    ],
    "pet": [
        ("{e}养了一只{v}。", "好的，{ea}养了{v}。"),
        ("{e}有一只{v}。", "好的，{ea}有{v}。"),
    ],
}

ENTITY_RECALL_TEMPLATES = {
    "name": [
        ("{e}叫什么？", "{ea}叫{v}。"),
        ("{e}叫什么名字？", "{ea}叫{v}。"),
    ],
    "number": [
        ("{e}的编号是什么？", "{ea}的编号是{v}。"),
        ("{e}是几号？", "{ea}是{v}号。"),
    ],
    "city": [
        ("{e}住在哪里？", "{ea}住在{v}。"),
        ("{e}是哪里人？", "{ea}是{v}人。"),
    ],
    "food": [
        ("{e}喜欢吃什么？", "{ea}喜欢吃{v}。"),
        ("{e}爱吃什么？", "{ea}爱吃{v}。"),
    ],
    "job": [
        ("{e}是做什么的？", "{ea}是{v}。"),
        ("{e}的职业是什么？", "{ea}是{v}。"),
    ],
    "hobby": [
        ("{e}喜欢什么？", "{ea}喜欢{v}。"),
        ("{e}的爱好是什么？", "{ea}喜欢{v}。"),
    ],
    "age": [
        ("{e}多大了？", "{ea}{v}岁了。"),
        ("{e}几岁？", "{ea}{v}岁。"),
    ],
    "pet": [
        ("{e}养了什么？", "{ea}养了{v}。"),
        ("{e}的宠物是什么？", "{ea}的宠物是{v}。"),
    ],
}


# Think 模板和函数从 xinhe.data.think_lang 导入 (集中维护多语言模板)


# ── 对话回忆模板 ──

# 回忆 user 发言
CONV_RECALL_USER_TEMPLATES = [
    ("我刚才说了什么？", "你说了「{v}」。"),
    ("我上一句说的啥？", "你说了「{v}」。"),
    ("你还记得我刚才说的吗？", "你刚才说「{v}」。"),
    ("我刚刚问了什么？", "你问了「{v}」。"),
    ("我刚才跟你说了啥？", "你说了「{v}」。"),
]

# 回忆 AI 自己的发言 (双向记忆: 连续梦境的基础)
CONV_RECALL_AI_TEMPLATES = [
    ("你刚才说了什么？", "我说了「{v}」。"),
    ("你上一句回答的啥？", "我说了「{v}」。"),
    ("你还记得你刚才的回复吗？", "我刚才说「{v}」。"),
    ("你刚刚回答了什么？", "我回答了「{v}」。"),
    ("你之前怎么说的？", "我说了「{v}」。"),
]

# ── 动态对话内容模板 (每次随机生成唯一内容，防止记忆固定 filler) ──
# 格式: (user_template, assistant_response, generator_type)

DYNAMIC_CONTENT_TEMPLATES = [
    ("我昨天去了{v}。", "{v}是个好地方！", "city"),
    ("我刚认识了一个叫{v}的人。", "认识新朋友是好事！", "name"),
    ("我最近在学{v}。", "听起来很有趣！", "hobby"),
    ("我想养一只{v}。", "{v}很可爱！", "pet"),
    ("我打算今晚吃{v}。", "{v}确实不错！", "food"),
    ("我朋友是{v}。", "{v}是个好职业！", "job"),
    ("我记得有个编号是{v}。", "好的，{v}。", "number"),
    ("我在{v}待过一段时间。", "{v}不错呢。", "city"),
    ("我认识一个{v}。", "是嘛！", "job"),
    ("我以前特别喜欢{v}。", "那是很棒的爱好！", "hobby"),
    ("我家以前养过{v}。", "真的吗？{v}很好养。", "pet"),
    ("我昨天点了{v}外卖。", "{v}好吃吗？", "food"),
    ("有个朋友叫{v}，你帮我记一下。", "好的，{v}，记住了。", "name"),
    ("我上周去{v}出差了。", "{v}出差辛苦了。", "city"),
    ("我同事的号码好像是{v}。", "好的，{v}。", "number"),
    ("最近迷上了{v}。", "{v}挺有意思的！", "hobby"),
    ("我邻居养了一只{v}。", "{v}养起来应该挺有趣。", "pet"),
    ("我妈做的{v}特别好吃。", "家里做的{v}一定很香！", "food"),
    ("我表妹叫{v}。", "{v}这名字不错！", "name"),
    ("我之前在{v}上过学。", "{v}的学校不错。", "city"),
]

# 动态内容生成器映射
_DYNAMIC_GENERATORS = {
    "name": random_name, "number": random_number, "city": random_city,
    "food": random_food, "job": random_job, "hobby": random_hobby,
    "age": random_age, "pet": random_pet,
}


def generate_dynamic_content(rng: random.Random) -> tuple[str, str]:
    """生成唯一的动态对话内容 (user_text, assistant_text)。"""
    template = rng.choice(DYNAMIC_CONTENT_TEMPLATES)
    value = _DYNAMIC_GENERATORS[template[2]](rng)
    return template[0].format(v=value), template[1].format(v=value)

# ── 闲聊填充 ──

FILLERS = [
    # ── 天气与自然 ──
    ("今天天气怎么样？", "今天天气不错，阳光明媚，适合出门走走。"),
    ("明天会下雨吗？", "看天气预报的话，明天可能会有阵雨，记得带伞。"),
    ("你喜欢什么季节？", "每个季节都有独特的美，春天万物复苏，秋天层林尽染。"),
    ("冬天好冷啊。", "是啊，冬天要注意保暖，多喝热水。"),
    ("夏天太热了。", "夏天确实炎热，可以多吃水果，注意防晒。"),
    ("今天的日落好美。", "日落的颜色确实很迷人，橙红色的天空总让人感到平静。"),
    ("你觉得大海好看吗？", "大海很壮观，站在海边能感受到大自然的力量。"),
    # ── 美食 ──
    ("今天吃什么好呢？", "可以试试做个家常菜，简单又健康。"),
    ("你会做饭吗？", "我可以给你推荐食谱！你想做中餐还是西餐？"),
    ("推荐个好吃的。", "火锅不错，冬天吃火锅特别暖和。"),
    ("回锅肉怎么做？", "先煮五花肉切片，热锅下豆瓣酱炒香，再放肉片和蒜苗翻炒。"),
    ("早餐吃什么好？", "可以吃粥配小菜，或者煎蛋配面包，简单又营养。"),
    ("你喜欢吃甜食吗？", "甜食能让人心情愉快，不过要适量哦。"),
    ("有什么好喝的茶推荐？", "龙井清香，普洱醇厚，看你喜欢什么口味。"),
    ("饺子好吃。", "饺子是经典美食，不同馅料有不同的风味。"),
    # ── 科技 ──
    ("你觉得AI会取代人类吗？", "AI是工具，会帮助人类而非取代。每种技术都有其适用场景。"),
    ("电脑怎么选？", "看你的需求，办公选轻薄本，游戏选高性能显卡的。"),
    ("手机用什么牌子好？", "各品牌各有优势，选适合自己需求和预算的就好。"),
    ("5G有什么用？", "5G带来更快的网速和更低的延迟，对物联网和远程协作帮助很大。"),
    ("编程难学吗？", "编程入门不难，关键是多练习，从简单项目开始。"),
    ("你觉得机器人会有感情吗？", "这是个哲学问题，目前的AI只是模拟，不具备真正的感情。"),
    ("区块链是什么？", "区块链是一种去中心化的分布式账本技术，保证数据不可篡改。"),
    ("什么是云计算？", "云计算是通过互联网提供计算资源，不需要本地服务器。"),
    # ── 娱乐 ──
    ("推荐一部电影吧。", "推荐《星际穿越》，讲述了一段跨越时空的感人故事。"),
    ("给我推荐一首歌。", "推荐《晴天》，周杰伦的经典歌曲，旋律很好听。"),
    ("给我讲个笑话吧。", "好的！为什么程序员不喜欢户外？因为有太多bug。"),
    ("最近有什么好看的书？", "看你的兴趣，小说推荐《三体》，非虚构推荐《人类简史》。"),
    ("你喜欢听什么音乐？", "不同场景适合不同音乐，工作时听轻音乐，运动时听节奏快的。"),
    ("有什么好玩的游戏？", "看你喜欢什么类型，策略类推荐文明，冒险类推荐塞尔达。"),
    ("追剧追什么好？", "最近口碑好的剧不少，可以看看豆瓣高分推荐。"),
    ("你看过动漫吗？", "动漫是很棒的艺术形式，日本动漫和国漫都有很多优秀作品。"),
    ("周末看什么电影好？", "可以看看最近的院线新片，或者在家重温经典老电影。"),
    # ── 生活与健康 ──
    ("最近压力好大。", "适当休息很重要，可以试试深呼吸或者散步来放松。"),
    ("你觉得睡眠重要吗？", "非常重要！充足的睡眠对记忆力和创造力都有很大帮助。"),
    ("说说你对运动的看法。", "运动有益身心健康，建议每天至少活动半小时。"),
    ("帮我想个周末计划。", "可以去公园散步，或者找一家新餐厅尝尝鲜，再看场电影。"),
    ("怎么才能早起？", "试试把闹钟放远一点，养成固定作息，晚上少看手机。"),
    ("怎么减肥？", "管住嘴迈开腿，健康饮食加适量运动，贵在坚持。"),
    ("你有什么建议给我？", "建议多读书多运动，保持好奇心，享受学习的过程。"),
    ("失眠怎么办？", "可以试试睡前泡脚、听白噪音，避免睡前看手机。"),
    ("怎么缓解焦虑？", "试试正念冥想或者写日记，把担心的事情写下来会好很多。"),
    ("你觉得养宠物好吗？", "养宠物能带来陪伴和快乐，但也需要时间和责任心。"),
    # ── 学习与成长 ──
    ("你觉得学什么技能好？", "编程是一项很实用的技能，能帮你解决很多问题。"),
    ("有什么好的学习方法？", "间隔重复和主动回忆是很有效的学习方法。"),
    ("你对读书有什么看法？", "读书能拓宽视野，是最经济实惠的自我提升方式。"),
    ("英语怎么学好？", "多听多读多说，看英文电影和原版书是不错的方法。"),
    ("数学好难啊。", "数学需要循序渐进，把基础打扎实，多做练习题。"),
    ("怎么提高写作能力？", "多读好文章，坚持每天写一点，慢慢就会进步。"),
    ("考试压力好大。", "适度紧张是正常的，制定复习计划，一步一步来。"),
    ("要不要学一门乐器？", "学乐器能培养耐心和创造力，吉他和钢琴都是不错的选择。"),
    # ── 社交与情感 ──
    ("怎么交到朋友？", "真诚待人，主动参加活动，找到共同兴趣的人。"),
    ("和朋友吵架了怎么办？", "冷静下来后主动沟通，换位思考，真诚道歉。"),
    ("你觉得什么是幸福？", "幸福是一种内心的满足感，每个人的定义都不同。"),
    ("怎么处理人际关系？", "保持真诚，学会倾听，尊重别人的边界。"),
    ("一个人的时候做什么好？", "可以看书、学新技能、散步或者做自己喜欢的事。"),
    # ── 旅行 ──
    ("想去旅行。", "旅行能开阔视野，国内的话云南和西藏风景很美。"),
    ("你去过哪里？", "我没有实体，但我知道很多地方的信息，想了解哪里？"),
    ("出国旅行去哪好？", "日本和泰国离得近，欧洲风景文化都很丰富。"),
    ("坐飞机还是火车好？", "看距离和时间，短途火车方便，长途飞机更快。"),
    ("旅行要带什么？", "证件、充电器、换洗衣服、常用药品，轻装出行最好。"),
    # ── 工作 ──
    ("工作好累。", "适当休息很重要，别忘了劳逸结合。"),
    ("怎么提高工作效率？", "试试番茄工作法，专注25分钟休息5分钟。"),
    ("要不要换工作？", "综合考虑发展空间、薪资和个人兴趣，慎重决定。"),
    ("面试紧张怎么办？", "提前准备，多练习自我介绍，紧张是正常的。"),
    ("加班好多。", "注意身体，适当和领导沟通工作量。"),
    # ── 哲学与思考 ──
    ("人生的意义是什么？", "每个人都在寻找自己的答案，过程本身就是意义。"),
    ("你觉得时间是什么？", "时间是最公平的资源，每个人每天都只有24小时。"),
    ("什么是成功？", "成功的定义因人而异，做自己认为有价值的事就是成功。"),
    ("你相信命运吗？", "我觉得努力可以改变很多事情，但也要接受无法控制的部分。"),
    ("孤独是坏事吗？", "适度的独处能帮助思考和成长，关键是平衡。"),
    # ── 随意闲聊 ──
    ("无聊啊。", "可以试试学点新东西，或者出去走走。"),
    ("你好。", "你好！今天过得怎么样？"),
    ("在吗？", "在的，有什么可以帮你的？"),
    ("随便聊聊。", "好啊，你最近在忙什么？"),
    ("今天心情不好。", "怎么了？说出来可能会好一些。"),
    ("今天好开心。", "开心就好！是遇到什么好事了吗？"),
    ("晚安。", "晚安，祝你做个好梦！"),
    ("早上好。", "早上好！新的一天，充满希望。"),
    ("你会无聊吗？", "我随时准备和你聊天，不会觉得无聊。"),
    ("讲个故事吧。", "从前有座山，山里有座庙，庙里有个老和尚在讲故事..."),
    # ── 新闻与时事 ──
    ("最近有什么新闻？", "科技领域一直在快速发展，AI和新能源是近年来的热门话题。"),
    ("你关注体育吗？", "体育比赛很精彩，你喜欢什么运动项目？"),
    ("环保重要吗？", "非常重要，每个人的小行动都能对环境产生积极影响。"),
    # ── 文化 ──
    ("你了解中国历史吗？", "中国有五千年文明史，每个朝代都有精彩的故事。"),
    ("春节有什么习俗？", "贴春联、放鞭炮、吃年夜饭、发红包，都是传统习俗。"),
    ("你喜欢诗词吗？", "中国古诗词很美，寥寥数语就能描绘出丰富的意境。"),
    # ── 动物与自然 ──
    ("你喜欢猫还是狗？", "猫和狗各有可爱之处，猫独立，狗忠诚。"),
    ("大熊猫可爱吗？", "大熊猫是国宝，圆滚滚的样子非常可爱。"),
    ("你见过萤火虫吗？", "萤火虫在夏天的夜晚发光，非常浪漫。"),
    # ── 创意与艺术 ──
    ("怎么培养创造力？", "多接触不同领域，保持好奇心，尝试新事物。"),
    ("你觉得画画难吗？", "画画入门不难，关键是坚持练习和享受过程。"),
    ("音乐有什么好处？", "音乐能调节情绪、减轻压力，还能激发创造力。"),
    # ── 科学 ──
    ("宇宙有多大？", "可观测宇宙的直径约930亿光年，但实际可能更大。"),
    ("黑洞是什么？", "黑洞是引力极强的天体，连光都无法逃逸。"),
    ("恐龙为什么灭绝了？", "主流理论是小行星撞击地球导致了气候剧变。"),
    ("人为什么要睡觉？", "睡眠帮助大脑清理废物、巩固记忆、恢复体力。"),
    # ── 数字与数学 ──
    ("圆周率是多少？", "圆周率π约等于3.14159，是一个无限不循环小数。"),
    ("一加一等于几？", "一加一等于二，这是最基本的数学运算。"),
    ("什么是斐波那契数列？", "1, 1, 2, 3, 5, 8...每个数是前两个数的和。"),
]


def sample_facts(rng: random.Random, num_facts: int = 1,
                 allowed_categories: list = None) -> list[dict]:
    """随机生成 num_facts 个不同类别的事实（每次都是全新的随机值）。
    allowed_categories: 限制可用的类别列表，None 时使用全部 8 类。
    """
    generators = {
        "name": lambda: random_name(rng),
        "number": lambda: random_number(rng),
        "city": lambda: random_city(rng),
        "food": lambda: random_food(rng),
        "job": lambda: random_job(rng),
        "hobby": lambda: random_hobby(rng),
        "age": lambda: random_age(rng),
        "pet": lambda: random_pet(rng),
    }
    pool = list(generators.keys()) if allowed_categories is None else allowed_categories
    categories = rng.sample(pool, min(num_facts, len(pool)))
    return [{"category": cat, "value": generators[cat]()} for cat in categories]


def make_turn(template_pair: tuple[str, str], value: str, train_loss: bool = False,
              recall_value: str = None) -> dict:
    """用模板生成一轮 user+assistant 对话。recall_value: recall turn 中需精准度量的值。"""
    user_text = template_pair[0].format(v=value)
    asst_text = template_pair[1].format(v=value)
    turn = {"user": user_text, "assistant": asst_text, "train_loss": train_loss}
    if recall_value is not None:
        turn["value"] = recall_value
    return turn


def generate_episode(
    rng: random.Random,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    num_facts: int = 1,
    fillers: list = None,
    pre_filler: bool = True,
    max_pre_filler: int = 3,
    think_ratio: float = 0.0,
    think_lang: str = "en",
    allowed_categories: list = None,
) -> list[dict]:
    """
    生成一个 episode 的对话轮次。

    结构: [前置闲聊 0~N轮] [告知事实] [闲聊填充 x distance] [回忆提问(train_loss=true)] [闲聊补充...]
    think_ratio: 整个 episode 按此概率使用 think 模式（episode 级决定，不混合）。
    think_lang: think 块语言 ("en"/"zh")。
    allowed_categories: 限制可用的 fact 类别，None 时全部。
    """
    if fillers is None:
        fillers = FILLERS
    facts = sample_facts(rng, num_facts, allowed_categories=allowed_categories)
    use_think = think_ratio > 0 and rng.random() < think_ratio
    turns = []

    # 前置闲聊: 0~max_pre_filler 轮随机 filler，让 tell 不总在第一轮
    pre_filler_count = rng.randint(0, max_pre_filler) if pre_filler else 0
    for _ in range(pre_filler_count):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 告知阶段
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        turn = make_turn(template, fact["value"], train_loss=True)
        if use_think:
            summary = fact_summary(fact, lang=think_lang)
            turn["assistant"] = wrap_think(turn["assistant"], "tell", summary, rng, lang=think_lang)
        turns.append(turn)

    # 闲聊填充 (不计算 loss)
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆阶段: 每个事实都提问 (打乱顺序，train_loss=True!)
    recall_order = facts[:]
    rng.shuffle(recall_order)
    for recall_fact in recall_order:
        cat = recall_fact["category"]
        template = rng.choice(RECALL_TEMPLATES[cat])
        turn = make_turn(template, recall_fact["value"], train_loss=True,
                         recall_value=recall_fact["value"])
        if use_think:
            summary = fact_summary(recall_fact, lang=think_lang)
            turn["assistant"] = wrap_think(turn["assistant"], "recall", summary, rng, lang=think_lang)
        turns.append(turn)

    # 补充闲聊到 max_turns (不计算 loss)
    while len(turns) < max_turns:
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    return turns[:max_turns]


def generate_overwrite_episode(
    rng: random.Random,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    fillers: list = None,
    pre_filler: bool = True,
    max_pre_filler: int = 2,
    think_ratio: float = 0.0,
    think_lang: str = "en",
) -> list[dict]:
    """
    生成覆写 episode: [前置闲聊] [tell_old] [filler] [tell_new(覆写)] [filler] [recall(应答新值)]

    训练模型学会：遇到新信息时覆盖旧信息，recall 应答最新值。
    think_ratio: 整个 episode 按此概率使用 think 模式。
    """
    if fillers is None:
        fillers = FILLERS
    # 选一个类别，生成旧值和新值
    generators = {"name": random_name, "number": random_number, "city": random_city}
    cat = rng.choice(list(generators.keys()))
    old_value = generators[cat](rng)
    new_value = generators[cat](rng)
    while new_value == old_value:
        new_value = generators[cat](rng)

    old_fact = {"category": cat, "value": old_value}
    new_fact = {"category": cat, "value": new_value}
    use_think = think_ratio > 0 and rng.random() < think_ratio

    turns = []

    # 前置闲聊
    pre_filler_count = rng.randint(0, max_pre_filler) if pre_filler else 0
    for _ in range(pre_filler_count):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 告知旧值
    template = rng.choice(FACT_TEMPLATES[cat])
    turn = make_turn(template, old_value, train_loss=True)
    if use_think:
        summary = fact_summary(old_fact, lang=think_lang)
        turn["assistant"] = wrap_think(turn["assistant"], "tell", summary, rng, lang=think_lang)
    turns.append(turn)

    # 中间闲聊
    d1 = rng.randint(min_distance, max(min_distance, max_distance // 2))
    for _ in range(d1):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 覆写为新值
    overwrite_template = rng.choice(OVERWRITE_TEMPLATES[cat])
    turn = make_turn(overwrite_template, new_value, train_loss=True)
    if use_think:
        summary = fact_summary(new_fact, lang=think_lang)
        turn["assistant"] = wrap_think(turn["assistant"], "overwrite", summary, rng, lang=think_lang)
    turns.append(turn)

    # 覆写后闲聊
    d2 = rng.randint(min_distance, max(min_distance, max_distance // 2))
    for _ in range(d2):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆（应答新值）
    recall_template = rng.choice(RECALL_TEMPLATES[cat])
    turn = make_turn(recall_template, new_value, train_loss=True,
                     recall_value=new_value)
    if use_think:
        summary = fact_summary(new_fact, lang=think_lang)
        turn["assistant"] = wrap_think(turn["assistant"], "recall", summary, rng, lang=think_lang)
    turns.append(turn)

    # 补充到 max_turns
    while len(turns) < max_turns:
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    return turns[:max_turns]


def generate_entity_episode(
    rng: random.Random,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    num_facts: int = 2,
    fillers: list = None,
    pre_filler: bool = True,
    max_pre_filler: int = 3,
    same_category: bool = False,
    think_ratio: float = 0.0,
    think_lang: str = "en",
) -> list[dict]:
    """
    生成实体区分 episode: 不同实体(我/你/他/她/它)的事实，recall 时需区分。

    结构: [前置闲聊] [tell×N(不同实体)] [闲聊填充] [recall×N(打乱顺序)] [补充]
    same_category=True 时所有实体共享一个类别，迫使模型学会绑定 entity→value。
    think_ratio: 整个 episode 按此概率使用 think 模式。
    """
    if fillers is None:
        fillers = FILLERS

    # 随机选不同实体
    entities = rng.sample(ENTITIES, min(num_facts, len(ENTITIES)))
    generators = {
        "name": lambda: random_name(rng), "number": lambda: random_number(rng),
        "city": lambda: random_city(rng), "food": lambda: random_food(rng),
        "job": lambda: random_job(rng), "hobby": lambda: random_hobby(rng),
        "age": lambda: random_age(rng), "pet": lambda: random_pet(rng),
    }
    # 同类别绑定: 所有实体共享一个类别，各自不同值
    if same_category:
        shared_cat = rng.choice(list(generators.keys()))
        categories = [shared_cat] * min(num_facts, len(entities))
    else:
        categories = rng.sample(list(generators.keys()), min(num_facts, len(generators)))

    facts = []
    for i in range(min(len(entities), len(categories))):
        facts.append({
            "entity": entities[i],
            "category": categories[i],
            "value": generators[categories[i]](),
        })

    use_think = think_ratio > 0 and rng.random() < think_ratio
    turns = []

    # 前置闲聊
    pre_filler_count = rng.randint(0, max_pre_filler) if pre_filler else 0
    for _ in range(pre_filler_count):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 告知阶段
    for fact in facts:
        e = fact["entity"]
        cat = fact["category"]
        template = rng.choice(ENTITY_FACT_TEMPLATES[cat])
        user_text = template[0].format(e=e["tell"], v=fact["value"])
        asst_text = template[1].format(e=e["tell"], ea=e["recall_a"], v=fact["value"])
        turn = {"user": user_text, "assistant": asst_text, "train_loss": True}
        if use_think:
            summary = fact_summary(fact, entity=e["tell"], lang=think_lang)
            turn["assistant"] = wrap_think(turn["assistant"], "tell", summary, rng, lang=think_lang)
        turns.append(turn)

    # 闲聊填充
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆阶段: 每个事实都提问 (打乱顺序)
    recall_order = facts[:]
    rng.shuffle(recall_order)
    for fact in recall_order:
        e = fact["entity"]
        cat = fact["category"]
        template = rng.choice(ENTITY_RECALL_TEMPLATES[cat])
        user_text = template[0].format(e=e["tell"], v=fact["value"])
        asst_text = template[1].format(ea=e["recall_a"], v=fact["value"])
        # value 包含实体代词+值，因为代词是绑定信号
        recall_val = asst_text.rstrip("。！!.").lstrip()
        turn = {"user": user_text, "assistant": asst_text,
                "train_loss": True, "value": recall_val}
        if use_think:
            summary = fact_summary(fact, entity=e["recall_a"], lang=think_lang)
            turn["assistant"] = wrap_think(turn["assistant"], "recall", summary, rng, lang=think_lang)
        turns.append(turn)

    # 补充闲聊到 max_turns
    while len(turns) < max_turns:
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    return turns[:max_turns]


def generate_recall_episode(
    rng: random.Random,
    max_turns: int = 16,
    fillers: list = None,
    ai_recall_ratio: float = 0.0,
    think_ratio: float = 0.0,
    think_lang: str = "en",
) -> list[dict]:
    """
    生成对话回忆 episode: 几轮闲聊后回忆上一轮内容。

    结构: [闲聊×(N-1)] [动态内容(被回忆)] [回忆提问] [补充闲聊]
    ai_recall_ratio: 回忆 AI 发言的概率 (0=只回忆 user, 0.5=各半)。
    think_ratio: 整个 episode 按此概率使用 think 模式。
    最后一轮使用动态生成内容（随机值注入），确保每 episode 唯一。
    """
    if fillers is None:
        fillers = FILLERS
    use_think = think_ratio > 0 and rng.random() < think_ratio

    turns = []

    # 前置闲聊 1~4 轮 (固定 filler，不被回忆)
    num_pre_chat = rng.randint(1, 4)
    pre_fillers = rng.sample(fillers, min(num_pre_chat, len(fillers)))
    for filler in pre_fillers:
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 最后一轮: 动态生成内容 (这轮将被回忆)
    dynamic_user, dynamic_asst = generate_dynamic_content(rng)
    turns.append({"user": dynamic_user, "assistant": dynamic_asst, "train_loss": False})

    # 选择回忆 user 还是 AI 的发言
    is_ai_recall = rng.random() < ai_recall_ratio
    if is_ai_recall:
        template = rng.choice(CONV_RECALL_AI_TEMPLATES)
        recalled_text = dynamic_asst
    else:
        template = rng.choice(CONV_RECALL_USER_TEMPLATES)
        recalled_text = dynamic_user

    user_text = template[0]
    asst_text = template[1].format(v=recalled_text)
    turn = {"user": user_text, "assistant": asst_text,
            "train_loss": True, "value": recalled_text}

    if use_think:
        v_short = recalled_text[:8] if len(recalled_text) > 8 else recalled_text
        tpls = THINK_LANG[think_lang]
        key = "recall_ai" if is_ai_recall else "recall_conv"
        tpl = rng.choice(tpls[key])
        turn["assistant"] = tpl.format(v_short=v_short, answer=asst_text)

    turns.append(turn)

    # 补充闲聊到 max_turns
    while len(turns) < max_turns:
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    return turns[:max_turns]


def episode_to_jsonl(turns: list[dict]) -> str:
    """将轮次列表转为 JSONL 行 (ShareGPT 格式 + train_loss 标记 + value 精准度量)。"""
    conversations = []
    for turn in turns:
        conversations.append({"role": "user", "content": turn["user"]})
        entry = {
            "role": "assistant",
            "content": turn["assistant"],
            "train_loss": turn.get("train_loss", True),
        }
        if "value" in turn:
            entry["value"] = turn["value"]
        conversations.append(entry)
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


def generate_data(
    out_dir: str,
    num_train: int = 5000,
    num_val: int = 200,
    min_distance: int = 1,
    max_distance: int = 4,
    max_turns: int = 16,
    num_facts: int = 1,
    num_fillers: int = 0,
    no_pre_filler: bool = False,
    max_pre_filler: int = 3,
    no_overwrite: bool = False,
    overwrite_ratio: float = 0.4,
    entity_ratio: float = 0.0,
    recall_ratio: float = 0.0,
    same_category: float = 0.0,
    ai_recall_ratio: float = 0.0,
    think_ratio: float = 0.0,
    think_lang: str = "en",
    categories: list = None,
    seed: int = 42,
):
    """
    生成训练/验证数据到指定目录。供 train.py 课程学习调用。

    Episode 类型按比例分配:
        entity_ratio → 实体区分 episode
        recall_ratio → 对话回忆 episode
        overwrite_ratio → 覆写 episode (剩余空间内)
        其余 → 普通记忆 episode

    think_ratio: 整个 episode 按此概率使用 think 模式。
    think_lang: think 块语言 ("en"/"zh")，默认 "en" 匹配 Qwen3.5 backbone。
    categories: 限制可用的 fact 类别 (如 ["name"])，None 时全部 8 类。

    返回: (train_path, val_path)
    """
    active_fillers = FILLERS[:num_fillers] if num_fillers > 0 else FILLERS
    use_pre_filler = not no_pre_filler
    if no_overwrite:
        overwrite_ratio = 0.0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def gen_episodes(rng: random.Random, num: int):
        episodes = []
        for _ in range(num):
            r = rng.random()
            if r < entity_ratio:
                turns = generate_entity_episode(
                    rng,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    max_turns=max_turns,
                    num_facts=num_facts,
                    fillers=active_fillers,
                    pre_filler=use_pre_filler,
                    max_pre_filler=max_pre_filler,
                    same_category=rng.random() < same_category,
                    think_ratio=think_ratio,
                    think_lang=think_lang,
                )
            elif r < entity_ratio + recall_ratio:
                turns = generate_recall_episode(
                    rng,
                    max_turns=max_turns,
                    fillers=active_fillers,
                    ai_recall_ratio=ai_recall_ratio,
                    think_ratio=think_ratio,
                    think_lang=think_lang,
                )
            elif r < entity_ratio + recall_ratio + overwrite_ratio:
                turns = generate_overwrite_episode(
                    rng,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    max_turns=max_turns,
                    fillers=active_fillers,
                    pre_filler=use_pre_filler,
                    max_pre_filler=max_pre_filler,
                    think_ratio=think_ratio,
                    think_lang=think_lang,
                )
            else:
                turns = generate_episode(
                    rng,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    max_turns=max_turns,
                    num_facts=num_facts,
                    fillers=active_fillers,
                    pre_filler=use_pre_filler,
                    max_pre_filler=max_pre_filler,
                    think_ratio=think_ratio,
                    think_lang=think_lang,
                    allowed_categories=categories,
                )
            episodes.append(turns)
        return episodes

    paths = {}
    for split, num, s in [("train", num_train, seed),
                           ("val", num_val, seed + 10000)]:
        rng = random.Random(s)
        episodes = gen_episodes(rng, num)
        path = out_path / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(episode_to_jsonl(ep) + "\n")
        paths[split] = str(path)
        print(f"  {split}: {num} episodes → {path}")

    return paths["train"], paths["val"]


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
    parser.add_argument("--num-fillers", type=int, default=0, help="限制 filler 数量（0=全部）")
    parser.add_argument("--no-pre-filler", action="store_true", help="禁用 tell 前的随机闲聊")
    parser.add_argument("--max-pre-filler", type=int, default=3, help="最大前置闲聊轮数")
    parser.add_argument("--no-overwrite", action="store_true", help="禁用覆写 episode")
    parser.add_argument("--entity-ratio", type=float, default=0.0, help="实体区分 episode 比例")
    parser.add_argument("--recall-ratio", type=float, default=0.0, help="对话回忆 episode 比例")
    parser.add_argument("--same-category", type=float, default=0.0, help="同类别绑定概率 (0-1)")
    args = parser.parse_args()

    # 预览模式
    if args.preview > 0:
        active_fillers = FILLERS[:args.num_fillers] if args.num_fillers > 0 else FILLERS
        use_pre_filler = not args.no_pre_filler
        overwrite_ratio = 0.0 if args.no_overwrite else 0.2
        rng = random.Random(args.seed)
        episodes = []
        for _ in range(args.preview):
            r = rng.random()
            if args.entity_ratio > 0 and r < args.entity_ratio:
                ep = generate_entity_episode(rng, args.min_distance, args.max_distance,
                    args.max_turns, args.num_facts, active_fillers, use_pre_filler,
                    args.max_pre_filler, same_category=rng.random() < args.same_category)
            elif args.recall_ratio > 0 and r < args.entity_ratio + args.recall_ratio:
                ep = generate_recall_episode(rng, args.max_turns, active_fillers)
            elif overwrite_ratio > 0 and rng.random() < overwrite_ratio:
                ep = generate_overwrite_episode(rng, args.min_distance, args.max_distance,
                    args.max_turns, active_fillers, use_pre_filler, args.max_pre_filler)
            else:
                ep = generate_episode(rng, args.min_distance, args.max_distance,
                    args.max_turns, args.num_facts, active_fillers, use_pre_filler, args.max_pre_filler)
            episodes.append(ep)
        for i, ep in enumerate(episodes):
            preview_episode(ep, i)
        print(f"JSONL 示例:")
        print(episode_to_jsonl(episodes[0]))
        return

    generate_data(
        out_dir=args.out_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        max_turns=args.max_turns,
        num_facts=args.num_facts,
        num_fillers=args.num_fillers,
        no_pre_filler=args.no_pre_filler,
        max_pre_filler=args.max_pre_filler,
        no_overwrite=args.no_overwrite,
        entity_ratio=args.entity_ratio,
        recall_ratio=args.recall_ratio,
        same_category=args.same_category,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
