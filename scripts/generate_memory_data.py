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
    fillers: list = None,
    pre_filler: bool = True,
    max_pre_filler: int = 3,
) -> list[dict]:
    """
    生成一个 episode 的对话轮次。

    结构: [前置闲聊 0~N轮] [告知事实] [闲聊填充 x distance] [回忆提问(train_loss=true)] [闲聊补充...]
    """
    if fillers is None:
        fillers = FILLERS
    facts = sample_facts(rng, num_facts)
    turns = []

    # 前置闲聊: 0~max_pre_filler 轮随机 filler，让 tell 不总在第一轮
    pre_filler_count = rng.randint(0, max_pre_filler) if pre_filler else 0
    for _ in range(pre_filler_count):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 告知阶段: 每个事实一轮 (计算 loss，让模型学会回复确认)
    for fact in facts:
        cat = fact["category"]
        template = rng.choice(FACT_TEMPLATES[cat])
        turn = make_turn(template, fact["value"], train_loss=True)
        turns.append(turn)

    # 闲聊填充 (不计算 loss)
    distance = rng.randint(min_distance, max_distance)
    for _ in range(distance):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆阶段: 随机挑一个事实提问 (train_loss=True!)
    recall_fact = rng.choice(facts)
    cat = recall_fact["category"]
    template = rng.choice(RECALL_TEMPLATES[cat])
    turn = make_turn(template, recall_fact["value"], train_loss=True)
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
) -> list[dict]:
    """
    生成覆写 episode: [前置闲聊] [tell_old] [filler] [tell_new(覆写)] [filler] [recall(应答新值)]

    训练模型学会：遇到新信息时覆盖旧信息，recall 应答最新值。
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

    turns = []

    # 前置闲聊
    pre_filler_count = rng.randint(0, max_pre_filler) if pre_filler else 0
    for _ in range(pre_filler_count):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 告知旧值
    template = rng.choice(FACT_TEMPLATES[cat])
    turns.append(make_turn(template, old_value, train_loss=True))

    # 中间闲聊
    d1 = rng.randint(min_distance, max(min_distance, max_distance // 2))
    for _ in range(d1):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 覆写为新值
    overwrite_template = rng.choice(OVERWRITE_TEMPLATES[cat])
    turns.append(make_turn(overwrite_template, new_value, train_loss=True))

    # 覆写后闲聊
    d2 = rng.randint(min_distance, max(min_distance, max_distance // 2))
    for _ in range(d2):
        filler = rng.choice(fillers)
        turns.append({"user": filler[0], "assistant": filler[1], "train_loss": False})

    # 回忆（应答新值）
    recall_template = rng.choice(RECALL_TEMPLATES[cat])
    turns.append(make_turn(recall_template, new_value, train_loss=True))

    # 补充到 max_turns
    while len(turns) < max_turns:
        filler = rng.choice(fillers)
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
    parser.add_argument("--num-fillers", type=int, default=0, help="限制 filler 数量（0=全部）")
    parser.add_argument("--no-pre-filler", action="store_true", help="禁用 tell 前的随机闲聊")
    parser.add_argument("--max-pre-filler", type=int, default=3, help="最大前置闲聊轮数")
    parser.add_argument("--no-overwrite", action="store_true", help="禁用覆写 episode")
    args = parser.parse_args()

    # filler 子集
    active_fillers = FILLERS[:args.num_fillers] if args.num_fillers > 0 else FILLERS
    use_pre_filler = not args.no_pre_filler
    overwrite_ratio = 0.0 if args.no_overwrite else 0.2

    out_dir = Path(args.out_dir)

    def gen_episodes(rng: random.Random, num: int):
        episodes = []
        for _ in range(num):
            if overwrite_ratio > 0 and rng.random() < overwrite_ratio:
                turns = generate_overwrite_episode(
                    rng,
                    min_distance=args.min_distance,
                    max_distance=args.max_distance,
                    max_turns=args.max_turns,
                    fillers=active_fillers,
                    pre_filler=use_pre_filler,
                    max_pre_filler=args.max_pre_filler,
                )
            else:
                turns = generate_episode(
                    rng,
                    min_distance=args.min_distance,
                    max_distance=args.max_distance,
                    max_turns=args.max_turns,
                    num_facts=args.num_facts,
                    fillers=active_fillers,
                    pre_filler=use_pre_filler,
                    max_pre_filler=args.max_pre_filler,
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
