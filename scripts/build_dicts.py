"""
词典与语料构建脚本。

两类素材统一管理：
  * 字符串字典 (.txt)：surnames / cities / foods / ...（12 类）
  * 对话语料 (.jsonl)：distract_chat / world_qa（2 类）

模式：
  --seed       用内置最小池子写所有素材到 xinhe/data/dicts/files/，零 API 成本。
               字符串类 ~50-200 条/类，语料类 ~10-20 pair/类。

  --expand     用 DeepSeek 把素材扩到 ≥ target：
                 字符串类目标 = --target（默认 1000）
                 语料类目标   = --target-pairs（默认 5000）
               逐类多轮调用、按 user / 实体去重、合并落盘。

  --version    重写 version.json 清单（含字符串类 + 语料类的 SHA1 + 切分统计）。

  --show       查看当前所有素材的 train/val/test 切分统计。

每个写入模式都会自动重写 version.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from xinhe.data.dicts import bank
from xinhe.data.dicts.bank import (
    FILES_DIR,
    VERSION_FILE,
    write_dict,
    write_pairs,
    write_version_manifest,
    load_bank,
    load_pairs,
)


# ────────────────────────────────────────────────────────────────────
# 配置区: 各类目标条数(可调整,调小 → 已有量 ≥ target 自动 skip)
# ────────────────────────────────────────────────────────────────────

# 字符串类目标(--target 仅作 fallback)
PER_CAT_TARGETS: dict[str, int] = {
    "surnames":      240,
    "given_names":   600,
    "cities":        400,
    "foods":         280,
    "jobs":          150,
    "hobbies":       200,
    "pets":          180,
    "colors":        200,
    "brands":        200,
    "organizations": 200,
    "project_codes": 1000,
    "passwords":     1000,
}

# 语料类目标(--target-pairs 仅作 fallback)
PER_NAME_TARGETS: dict[str, int] = {
    "distract_chat": 4000,
    "world_qa":      4000,
}


# ── seed mode：内置最小池子 ──
# 这里仅放每类的初始种子（50-200 条），DeepSeek expand 阶段会扩到 1000+。

SEED_DATA: dict[str, list[str]] = {}

SEED_DATA["surnames"] = [
    "赵","钱","孙","李","周","吴","郑","王","冯","陈","褚","卫","蒋","沈","韩","杨",
    "朱","秦","尤","许","何","吕","施","张","孔","曹","严","华","金","魏","陶","姜",
    "戚","谢","邹","喻","柏","水","窦","章","云","苏","潘","葛","奚","范","彭","郎",
    "鲁","韦","昌","马","苗","凤","花","方","俞","任","袁","柳","酆","鲍","史","唐",
    "费","廉","岑","薛","雷","贺","倪","汤","滕","殷","罗","毕","郝","邬","安","常",
    "乐","于","时","傅","皮","卞","齐","康","伍","余","元","卜","顾","孟","平","黄",
    "和","穆","萧","尹","姚","邵","湛","汪","祁","毛","禹","狄","米","贝","明","臧",
    "计","伏","成","戴","谈","宋","茅","庞","熊","纪","舒","屈","项","祝","董","梁",
    "杜","阮","蓝","闵","席","季","麻","强","贾","路","娄","危","江","童","颜","郭",
    "梅","盛","林","刁","钟","徐","邱","骆","高","夏","蔡","田","樊","胡","凌","霍",
    "虞","万","支","柯","昝","管","卢","莫","经","房","裘","缪","干","解","应","宗",
    "丁","宣","贲","邓","郁","单","杭","洪","包","诸","左","石","崔","吉","钮","龚",
    "程","嵇","邢","滑","裴","陆","荣","翁",
    "欧阳","司马","上官","诸葛","东方","公孙","慕容","皇甫",
    "令狐","独孤","南宫","西门","百里","呼延","端木","轩辕",
]

SEED_DATA["given_names"] = [
    "伟","芳","娜","秀英","敏","静","丽","强","磊","洋","勇","艳","杰","娟","涛","明",
    "超","兰","霞","平","刚","桂","文","辉","玲","华","红","军","燕","萍","建","春",
    "琴","云","飞","峰","凤","林","鑫","波","健","彬","斌","宇","浩","然","博","宏",
    "志","海","岩","鹏","旭","俊","哲","睿","翔","晨","辰","阳","凯","昊","龙","瑞",
    "雪","梅","莹","倩","颖","琳","璐","薇","婷","欣","悦","妍","佳","雨","思","涵",
    "蕊","馨","怡","诗","梦","宁","晴","瑶","萌","洁","蓉","露","菲","寒","冰","月",
    "星","风","晓","天","正","德","义","礼","智","信","仁","勤","和","安","泰","康",
    "裕","福","祥","荣","昌","盛","兴","国","栋","良","成","光","达","永","长","新",
    "胜","学","才","松","柏","茂","进","舟","帆","恒","毅","豪","远","航","程","锦",
    "绣","昕","彤","曦","妮","璇","琪","萱","蓓","蕾","苒","葵","茜","莲","竹",
    "Alice","Bob","Charlie","David","Emma","Frank","Grace","Henry","Iris","Jack",
    "Kate","Leo","Mia","Noah","Olivia","Paul","Quinn","Ruby","Sam","Tom",
    "Uma","Vera","Will","Xena","Yuki","Zoe","Alex","Luna","Max","Lily",
]

SEED_DATA["cities"] = [
    "北京","上海","广州","深圳","成都","杭州","武汉","西安","南京","重庆","苏州","长沙",
    "青岛","厦门","昆明","大连","天津","沈阳","哈尔滨","长春","济南","郑州","石家庄","太原",
    "合肥","福州","南昌","兰州","贵阳","南宁","海口","银川","西宁","呼和浩特","乌鲁木齐","拉萨",
    "无锡","常州","徐州","扬州","南通","镇江","泰州","盐城","淮安","宿迁","连云港",
    "宁波","温州","嘉兴","湖州","绍兴","金华","台州","丽水","衢州","舟山",
    "芜湖","蚌埠","淮南","马鞍山","铜陵","安庆","黄山","滁州","阜阳","宿州","亳州","池州","宣城","六安",
    "烟台","潍坊","临沂","淄博","济宁","泰安","威海","日照","德州","聊城","滨州","菏泽","枣庄","东营",
    "珠海","佛山","东莞","中山","惠州","江门","湛江","茂名","肇庆","汕头","揭阳","梅州","韶关","清远","河源","阳江","潮州","云浮",
    "洛阳","开封","平顶山","安阳","鹤壁","新乡","焦作","濮阳","许昌","漯河","三门峡","南阳","商丘","信阳","周口","驻马店",
    "桂林","柳州","北海","梧州","钦州","百色","玉林","贺州",
    "株洲","湘潭","衡阳","邵阳","岳阳","常德","张家界","益阳","郴州","永州","怀化","娄底",
    "遵义","六盘水","安顺","铜仁","曲靖","玉溪","保山","昭通","丽江","临沧","大理","德宏",
    "宜宾","绵阳","德阳","南充","乐山","泸州","达州","遂宁","内江","自贡","广元","眉山","攀枝花","雅安","巴中","资阳",
    "咸阳","宝鸡","渭南","延安","汉中","榆林","安康","商洛","大同","阳泉","长治","晋城","朔州","晋中","运城","忻州","临汾","吕梁",
    "唐山","秦皇岛","邯郸","邢台","保定","张家口","承德","廊坊","衡水","沧州",
    "鞍山","抚顺","本溪","丹东","锦州","营口","阜新","辽阳","盘锦","铁岭","朝阳","葫芦岛",
    "吉林","四平","辽源","通化","白山","松原","白城","延边",
    "齐齐哈尔","牡丹江","佳木斯","大庆","鸡西","双鸭山","伊春","七台河","鹤岗","黑河","绥化",
    "漳州","泉州","三明","莆田","南平","龙岩","赣州","吉安","宜春","抚州","上饶","景德镇","萍乡","新余","鹰潭",
]

SEED_DATA["foods"] = [
    "火锅","烤鸭","麻辣烫","饺子","拉面","小龙虾","酸菜鱼","宫保鸡丁","麻婆豆腐","鱼香肉丝",
    "煎饼","包子","馄饨","凉皮","肉夹馍","臭豆腐","螺蛳粉","热干面","刀削面",
    "三明治","牛排","意大利面","咖喱饭","炸鸡","寿司","披萨",
    "红烧肉","清蒸鲈鱼","水煮鱼","糖醋排骨","麻辣香锅","蒜蓉粉丝","葱爆牛肉","干煸豆角",
    "酱焖鸡","香煎三文鱼","清蒸虾","烤羊排","卤肉饭","凉拌黄瓜","油炸花生","炖排骨",
    "红烧茄子","清炒西兰花","水煮肉片","糖醋里脊","麻辣豆腐","蒜蓉空心菜","葱爆羊肉",
    "干煸四季豆","酱焖排骨","香煎鸡腿","清蒸石斑鱼","烤翅","卤味拼盘","凉拌木耳",
    "炖牛腩","酸辣土豆丝","麻婆豆腐煲","蚂蚁上树","回锅肉","水煮牛肉","辣子鸡","口水鸡",
    "白切鸡","盐焗鸡","叫花鸡","北京烤鸭","片皮鸭","东坡肉","狮子头","酱牛肉",
    "鸳鸯火锅","牛肉面","羊肉串","小笼包","生煎包","肠粉","煲仔饭","云吞面","鸡蛋灌饼",
    "豆浆油条","胡辣汤","羊肉泡馍","biangbiang面","臊子面","油泼面","炸酱面",
    "重庆小面","担担面","葱油拌面","阳春面","片儿川","炒河粉","沙茶面","卤面","荞麦面",
    "粢饭团","八宝饭","糯米鸡","驴打滚","艾窝窝","糖油饼","南瓜饼","春卷","糍粑",
]

SEED_DATA["jobs"] = [
    "程序员","教师","医生","律师","设计师","工程师","记者","厨师","警察","消防员",
    "护士","会计","司机","建筑师","摄影师","作家","画家","音乐家","导演","演员",
    "销售","翻译","研究员","飞行员","快递员","外卖员","理发师","园丁","农民","渔民",
    "木匠","电工","水管工","保安","服务员","收银员","清洁工","保洁员","保姆","月嫂",
    "钢琴老师","舞蹈老师","健身教练","瑜伽老师","心理咨询师","营养师","按摩师","美容师",
    "美甲师","纹身师","插画师","UI设计师","UX设计师","产品经理","项目经理","运营经理",
    "市场经理","HR专员","行政助理","秘书","前台","导购","柜员","信贷员","保险经纪",
    "证券分析师","投资顾问","税务师","审计师","公证员","法务","知识产权代理人","专利律师",
    "数据分析师","算法工程师","前端工程师","后端工程师","全栈工程师","测试工程师","DevOps",
    "运维工程师","DBA","架构师","CTO","CEO","COO","创始人","合伙人",
    "兽医","宠物美容师","驯犬师","实验员","流行病学家","公卫医师","药剂师","中医","针灸师",
    "牙医","眼科医生","儿科医生","急诊医生","外科医生","内科医生","麻醉师","放射科医生",
    "记者","主持人","播音员","编剧","制片人","摄像","剪辑师","调音师","灯光师","化妆师",
    "造型师","服装设计师","珠宝设计师","工业设计师","景观设计师","室内设计师","平面设计师",
]

SEED_DATA["hobbies"] = [
    "打篮球","踢足球","游泳","跑步","爬山","骑自行车","打羽毛球","打乒乓球","下棋","钓鱼",
    "画画","弹吉他","弹钢琴","唱歌","看电影","读书","写作","摄影","旅行","做饭",
    "养花","打游戏","滑雪","冲浪","瑜伽","跳舞","书法","编程","种菜","看动漫",
    "听音乐","看话剧","看歌剧","看演唱会","看脱口秀","看相声","看综艺","刷短视频","写博客","做播客",
    "练书法","练国画","练油画","练水彩","捏陶","做手工","做木工","做模型","做拼图","玩魔方",
    "练瑜伽","练普拉提","练太极","练咏春","练散打","练拳击","练击剑","练射箭","攀岩","蹦极",
    "潜水","浮潜","冲浪","桨板","皮划艇","帆船","风筝冲浪","滑板","轮滑","街舞",
    "拉丁","民族舞","芭蕾","现代舞","街头戏剧","脱口秀","桌游","剧本杀","密室逃脱","电竞",
    "捡漏古玩","收藏邮票","收藏徽章","收藏明信片","集邮","集币","养鱼","养鸟","养猫","养狗",
    "种多肉","插花","茶艺","咖啡拉花","调酒","品酒","品茶","品咖啡","烘焙","做甜点",
    "做咖啡","做奶茶","做寿司","做意面","做韩餐","做日料","做西餐","研究菜谱","逛菜市场","逛超市",
    "逛博物馆","逛美术馆","逛动物园","逛植物园","逛公园","逛书店","逛二手书店","逛集市","逛夜市","看星星",
]

SEED_DATA["pets"] = [
    "猫","狗","兔子","仓鼠","鹦鹉","乌龟","金鱼","柯基","泰迪","柴犬",
    "英短","布偶猫","边牧","拉布拉多","哈士奇","比熊","蓝猫","暹罗猫","龙猫","刺猬",
    "金毛","萨摩耶","贵宾犬","博美","吉娃娃","马尔济斯","雪纳瑞","秋田犬","阿拉斯加","大丹犬",
    "罗威纳","杜宾","德牧","古牧","牛头梗","柯尔鸭","巴哥","八哥","斗牛犬","西高地",
    "美短","橘猫","三花猫","狸花猫","奶牛猫","加菲猫","波斯猫","布偶","缅因猫","折耳猫",
    "无毛猫","土猫","奶牛犬","小奶猫","小奶狗","小奶兔","小奶鼠","龙鱼","锦鲤","孔雀鱼",
    "斗鱼","神仙鱼","清道夫","虎皮鱼","红鹦鹉","蓝鹦鹉","百灵","画眉","八哥鸟","金丝雀",
    "玄凤","虎皮鹦鹉","和尚鹦鹉","太阳锥尾","小太阳","折衷","金刚","非洲灰","凯克","金太阳",
    "巴西龟","草龟","鳄龟","乌龟","陆龟","水龟","蜥蜴","壁虎","守宫","角蛙",
    "牛蛙","蝾螈","水蜥","蛇","蟒蛇","球蟒","玉米蛇","王锦蛇","树蛙","树蜥",
    "兜虫","锹甲","蝎子","蜘蛛","狼蛛","捕鸟蛛","蚕宝宝","蜜蜂","蚂蚁","蜗牛",
    "豚鼠","龙猫","蜜袋鼯","土拨鼠","松鼠","鼯鼠","黄金鼠","花栗鼠","兔子","荷兰猪",
]

SEED_DATA["colors"] = [
    "红色","橙色","黄色","绿色","蓝色","靛蓝","紫色","粉色","白色","黑色","灰色","棕色",
    "深红","暗红","枣红","酒红","玫瑰红","桃红","樱桃红","胭脂红","赤色","朱红","橘红",
    "茜红","赭红","品红","洋红","嫣红","粉红","蜜桃粉","婴儿粉","裸粉","灰粉","奶粉色",
    "鲑鱼红","珊瑚红","玛瑙红","橘黄","金黄","琥珀色","土黄","卡其黄","奶黄","蛋黄","姜黄",
    "菜花黄","小麦色","米色","奶油色","象牙白","乳白","骨白","珍珠白","雪白","蛋壳白",
    "暖白","冷白","原色","原木色","咖色","深咖","浅咖","摩卡","榛子色","焦糖色","巧克力色",
    "可可色","拿铁色","驼色","卡其","土色","棕褐","棕黄","赭石","赭色","褐色",
    "豆沙色","暗驼","薄荷绿","抹茶绿","荧光绿","军绿","橄榄绿","墨绿","松针绿","祖母绿",
    "翠绿","葱绿","湖绿","苹果绿","薄荷青","青色","青蓝","蔚蓝","湖蓝","海蓝","深海蓝",
    "孔雀蓝","宝石蓝","群青","钴蓝","靛蓝","藏蓝","海军蓝","牛仔蓝","天蓝","雾蓝","蓝灰",
    "丁香紫","薰衣草紫","葡萄紫","紫罗兰","茄紫","深紫","淡紫","粉紫","酒红紫","莓紫",
    "黛紫","藕色","藕粉","裸色","肤色","奶茶色","抹茶奶绿","脏粉","脏橘","落日橘","暮光紫",
    "夜空蓝","深空灰","炭灰","烟灰","银灰","水泥灰","鸽子灰","雾霾灰","哑光黑","纯黑","墨黑",
]

SEED_DATA["brands"] = [
    "苹果","三星","华为","小米","OPPO","vivo","荣耀","联想","戴尔","惠普",
    "宝马","奔驰","奥迪","保时捷","法拉利","兰博基尼","大众","丰田","本田","日产",
    "特斯拉","比亚迪","蔚来","小鹏","理想","哪吒","零跑","岚图","极氪","问界",
    "可口可乐","百事可乐","雪碧","芬达","王老吉","加多宝","康师傅","统一","农夫山泉","怡宝",
    "百威","青岛","雪花","哈尔滨","乐堡","科罗娜","喜力","嘉士伯","燕京","纯生",
    "茅台","五粮液","泸州老窖","汾酒","剑南春","水井坊","郎酒","古井贡","洋河","西凤酒",
    "阿迪达斯","耐克","彪马","新百伦","李宁","安踏","特步","361度","回力","飞跃",
    "宜家","无印良品","名创优品","迪卡侬","ZARA","H&M","优衣库","GAP","NEXT","COS",
    "肯德基","麦当劳","汉堡王","必胜客","赛百味","DQ","哈根达斯","乐事","奥利奥","旺旺",
    "雀巢","星巴克","瑞幸","Manner","M Stand","Tims","Costa","上岛咖啡","太平洋咖啡","85度C",
    "海底捞","呷哺呷哺","凑凑","小龙坎","大龙燚","蜀大侠","巴奴","谭鸭血","海鲜大咖","西贝",
    "腾讯","阿里巴巴","字节跳动","美团","京东","拼多多","百度","网易","携程","滴滴",
    "飞猪","去哪儿","同程","高德","支付宝","微信","抖音","快手","小红书","B站",
]

SEED_DATA["organizations"] = [
    "联合国","世界银行","世卫组织","世贸组织","联合国教科文组织","国际刑警","北约","欧盟",
    "金砖国家","二十国集团","上海合作组织","东盟","非盟","阿盟","欧佩克","奥林匹克委员会",
    "国际奥委会","国际足联","亚足联","欧足联","NBA","CBA","中超","英超","西甲","德甲","意甲",
    "中科院","清华大学","北京大学","复旦大学","浙江大学","上海交大","南京大学","武汉大学","中山大学","厦门大学",
    "哈工大","同济大学","北航","北邮","西交大","华中科大","北师大","北外","上外","对外经贸",
    "麻省理工","哈佛","耶鲁","斯坦福","普林斯顿","康奈尔","哥伦比亚大学","伯克利","加州理工","卡内基梅隆",
    "牛津","剑桥","帝国理工","伦敦政经","UCL","爱丁堡","曼彻斯特","布里斯托","华威","KCL",
    "国家自然基金委","科技部","教育部","工信部","商务部","外交部","卫健委","公安部","住建部","人社部",
    "财政部","税务总局","海关总署","央行","证监会","银保监会","外汇局","审计署","国资委","发改委",
    "中国移动","中国联通","中国电信","中国广电","国家电网","南方电网","中石油","中石化","中海油","中核",
    "中国邮政","中国铁建","中国中铁","中交建","中建集团","中冶集团","中粮集团","中粮","华润","招商局",
    "中信","光大","中投","中投公司","国投","保利","中粮","五矿","兵装","兵器",
    "腾讯","阿里巴巴","百度","字节跳动","京东","美团","拼多多","滴滴","快手","小米",
    "Google","Apple","Microsoft","Amazon","Meta","Netflix","Tesla","NVIDIA","AMD","Intel",
    "OpenAI","Anthropic","DeepMind","Hugging Face","Mistral","Cohere","xAI","Stability AI","Runway","Midjourney",
]

SEED_DATA["project_codes"] = [
    "K9Q-27","M3X-08","R7B-15","T2D-44","P5N-93","Z8L-12","V4C-67","X6H-31",
    "A1B-22","B2C-33","C3D-44","D4E-55","E5F-66","F6G-77","G7H-88","H8I-99",
    "AX-101","BX-202","CX-303","DX-404","EX-505","FX-606","GX-707","HX-808",
    "Q1-07","Q2-14","Q3-21","Q4-28","Q5-35","Q6-42","Q7-49","Q8-56",
    "RX-1","RX-2","RX-3","RX-4","RX-5","SX-9","SX-19","SX-29","SX-39","SX-49",
    "Project-Alpha","Project-Beta","Project-Gamma","Project-Delta","Project-Epsilon","Project-Zeta","Project-Eta","Project-Theta",
    "ARK-1","ARK-2","ARK-3","ORION-7","ORION-8","NEPTUNE-3","JUPITER-9","SATURN-5","MARS-2","VENUS-1",
    "HX-21A","HX-22B","HX-23C","HX-24D","HX-25E","HX-26F","HX-27G","HX-28H",
    "Z-001","Z-002","Z-003","Z-004","Z-005","Z-006","Z-007","Z-008","Z-009","Z-010",
    "TF-19","TF-20","TF-21","TF-22","TF-23","TF-24","TF-25","TF-26","TF-27","TF-28",
    "BLU-12","RED-08","GRN-44","YEL-77","PRP-31","ORG-99","CYN-55","MGN-66",
    "α-12","β-7","γ-31","δ-66","ε-101","ζ-22","η-19","θ-88","λ-77","μ-44","σ-55",
]

SEED_DATA["passwords"] = [
    "白桦林","北极星","春之声","东风破","风车镇","古井深","海蓝时","红玫瑰",
    "金沙滩","蓝鲸湾","落日塔","明月夜","南极光","暖冬日","破晓时","青松林",
    "瑞雪飞","三叶草","太阳花","乌篷船","西风渡","小桥流","旭日升","月光华",
    "破晓","暗夜","幽兰","千寻","北辰","南山","东海","西楼","归途","逐光",
    "孤岛","漫步","云端","深海","星辉","流年","素笺","清欢","余晖","残阳",
    "春溪","夏阳","秋月","冬雪","梅雨","梨花","樱时","枫眠","萤火","萍踪",
    "Banana42","Mango77","Apple19","Pear88","Cherry07","Peach33","Lemon54","Lime21","Grape66","Melon09",
    "蓝色钢琴","绿色窗台","红色风车","金色稻浪","紫色铃兰","白色海鸥","橙色月亮","黑色玫瑰",
    "ButterFly99","DragonFly07","Phoenix12","Eagle33","Shark77","Tiger44","Lion88","Wolf21",
    "晨曦微光","暮色四合","残月如钩","新雪初霁","老树新枝","旧梦重温","薄雾浓云","清风徐来",
    "Falcon-9","Hawk-7","Crow-12","Sparrow-3","Owl-22","Raven-31","Swan-8","Swallow-5",
    "白马湖畔","梧桐巷口","樱木道前","银杏路尾","桂花街角","松林深处","竹林书屋","枫叶山庄",
    "Sirius21","Vega07","Altair12","Polaris33","Capella44","Procyon77","Rigel88","Antares99",
    "雾色山岚","枯山水","流云镜","半山亭","深巷醉","旧时光","老唱片","褪色照片","泛黄日记","蓝调夜",
    "BlueRain88","RedDawn22","GoldDusk33","SilverFog44","BlackLake55","WhiteHill66","GreenSea77","PurpleSky99",
    "竹影摇","风过松","雪落梅","雨打芭蕉","月照轩","云栖谷","松涛阵","泉水叮咚","小桥流水","古道西风",
]


# ── Pair 类 seed：对话语料种子（少量，主要靠 expand）──

SEED_PAIRS: dict[str, list[dict]] = {}

SEED_PAIRS["distract_chat"] = [
    {"user": "最近天气真冷啊。", "assistant": "是啊，记得多穿件外套别感冒了。"},
    {"user": "周末有什么好去处推荐吗？", "assistant": "可以去公园散散步，或者找家咖啡馆窝一下午。"},
    {"user": "今天好困，咖啡都救不了。", "assistant": "估计是没睡好，午休补个眠效果最快。"},
    {"user": "感觉时间过得好快。", "assistant": "确实，眨眼就一周过去了，要珍惜当下。"},
    {"user": "工作压力有点大。", "assistant": "可以适当放松一下，听点轻音乐或者出去走走。"},
    {"user": "晚上吃什么好呢？", "assistant": "天冷可以来碗热汤面，简单又暖胃。"},
    {"user": "你看最近那个新剧了吗？", "assistant": "听朋友提过，节奏紧凑画面也精致，找时间想追一下。"},
    {"user": "最近健身有效果吗？", "assistant": "坚持下来体感好多了，关键是规律饮食配合。"},
    {"user": "下班路上堵得不行。", "assistant": "高峰期老问题了，能错峰就尽量错峰吧。"},
    {"user": "猫咪今天又拆家了。", "assistant": "哈哈太典型了，给它准备点磨爪玩具能稍微转移注意力。"},
    {"user": "明天要早起开会，好烦。", "assistant": "提前定两个闹钟，今晚早点睡补觉。"},
    {"user": "最近想换个发型。", "assistant": "短发清爽，长发耐看，看你日常打理时间多不多。"},
    {"user": "总觉得手机内存不够用。", "assistant": "可以先清理下相册和缓存，实在不行就考虑换机。"},
    {"user": "新买的耳机音质不错。", "assistant": "戴着舒服比啥都重要，长时间听也不累。"},
    {"user": "今天阳光很好。", "assistant": "适合出去晒晒太阳，对心情和身体都好。"},
    {"user": "在外面吃饭还是自己做？", "assistant": "工作日图省事点外卖，周末慢慢做点喜欢的菜。"},
    {"user": "最近看了本书挺有意思。", "assistant": "好书能让人沉下来，多读几遍体会更深。"},
    {"user": "假期有什么计划？", "assistant": "不想跑远，打算在家附近的小镇逛逛，散散心。"},
    {"user": "买了双新跑鞋。", "assistant": "新鞋走路都精神，记得磨合一下再跑长距离。"},
    {"user": "感觉最近网速好慢。", "assistant": "重启路由器试试，不行就联系下运营商查线路。"},
]

SEED_PAIRS["world_qa"] = [
    {"user": "金字塔主要位于哪个国家？", "assistant": "金字塔主要位于埃及，特别是吉萨高原地区。"},
    {"user": "光速是多少？", "assistant": "光在真空中的速度约为每秒 299,792 公里。"},
    {"user": "地球到太阳的平均距离是多少？", "assistant": "约 1.496 亿公里，也就是 1 个天文单位（AU）。"},
    {"user": "DNA 由几种碱基组成？", "assistant": "四种：腺嘌呤（A）、胸腺嘧啶（T）、胞嘧啶（C）、鸟嘌呤（G）。"},
    {"user": "莎士比亚是哪个国家的作家？", "assistant": "英国，出生于 16 世纪后期，代表作有《哈姆雷特》《罗密欧与朱丽叶》等。"},
    {"user": "牛顿三大定律分别是什么？", "assistant": "惯性定律、加速度定律（F=ma）、作用与反作用定律。"},
    {"user": "中国四大发明是什么？", "assistant": "造纸术、印刷术、火药、指南针。"},
    {"user": "人体有多少块骨头？", "assistant": "成年人通常有 206 块，新生儿则更多，会随发育合并。"},
    {"user": "水的化学式是什么？", "assistant": "H2O，由两个氢原子和一个氧原子组成。"},
    {"user": "氧气在大气中的占比大约多少？", "assistant": "约 21%，氮气约占 78%，其余是惰性气体和二氧化碳等。"},
    {"user": "互联网的前身叫什么？", "assistant": "ARPANET，由美国国防部高级研究计划局（DARPA）在 1969 年建立。"},
    {"user": "圆周率 π 大约是多少？", "assistant": "约 3.14159，是一个无理数，无限不循环小数。"},
    {"user": "世界上最高的山峰是？", "assistant": "珠穆朗玛峰，海拔约 8848.86 米，位于中国与尼泊尔交界。"},
    {"user": "阿尔卑斯山在哪些国家？", "assistant": "横跨法国、瑞士、意大利、奥地利、德国、列支敦士登、斯洛文尼亚等。"},
    {"user": "诺贝尔奖有几大类？", "assistant": "物理学、化学、生理学或医学、文学、和平、经济学（后增设）。"},
    {"user": "二战开始与结束的年份？", "assistant": "1939 年开始，1945 年结束。"},
    {"user": "心脏每分钟大约跳多少次？", "assistant": "成人安静时约 60-100 次。"},
    {"user": "钢琴有多少个琴键？", "assistant": "标准钢琴有 88 个键：52 白键 + 36 黑键。"},
    {"user": "光合作用主要发生在植物的哪里？", "assistant": "叶片中的叶绿体里，主要利用阳光、水和二氧化碳合成糖。"},
    {"user": "地球的自转周期是多久？", "assistant": "约 23 小时 56 分钟（恒星日），日常使用的是 24 小时太阳日。"},
]


# ── Modes ──

def cmd_seed() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[seed] 写入 {FILES_DIR}")
    for category, entries in SEED_DATA.items():
        n = write_dict(category, entries)
        print(f"  [str] {category}: {n} 条")
    for name, pairs in SEED_PAIRS.items():
        n = write_pairs(name, pairs)
        print(f"  [pair] {name}: {n} 条")
    manifest = write_version_manifest()
    print(f"[seed] dict_version={manifest['dict_version']}, "
          f"categories={len(manifest['categories'])}, corpora={len(manifest['corpora'])}")


from xinhe.data.samplers import resolve as _resolve_sampler


def _expand_strings(target: int, model: str) -> None:
    """字符串字典扩充。target 仅作 fallback；具体 per-category target 见 PER_CAT_TARGETS。"""
    call_with_retry, ApiError = _resolve_sampler(model)

    PROMPTS = {
        "surnames": "中文姓氏（单字 / 复姓），保持简体中文。",
        "given_names": "中文常见名（不含姓氏，1-2 字），男女混合。也可少量含英文 / 昵称。",
        "cities": "城市名称：约 70-80% 中国大陆（直辖市/省会/地级市/常见县级市），约 20-30% 外国主要城市（如东京、纽约、伦敦、首尔、新加坡、悉尼等）。",
        "foods": "中式菜名 + 小吃 + 主食。包含川菜、粤菜、鲁菜、淮扬等多菜系。",
        "jobs": "现代职业 / 岗位 / 工种，覆盖白领、蓝领、专业人士、自由职业。",
        "hobbies": "兴趣爱好 / 业余活动，覆盖运动、艺术、社交、科技、生活。",
        "pets": "宠物名称（不是宠物的名字，是物种名），包含猫犬种类、鸟、爬宠等。",
        "colors": "颜色细粒度名（如雾蓝色、雪松绿），可包含通俗色名 + 文艺色名。",
        "brands": "品牌名（中英混合），覆盖科技、汽车、食品、服饰、餐饮、互联网。",
        "organizations": "公司 / 机构 / 政府部门 / 大学 / 国际组织名。",
        "project_codes": "字母+数字代号，形如 K9Q-27 / RX-101 / Project-Alpha / α-12。",
        "passwords": "5-12 个字符的暗号短语，可以是中文意象（白桦林）/ 英文 + 数字（Falcon-9）/ 古诗化。",
    }

    # per-category target 见模块顶部 PER_CAT_TARGETS

    for category, prompt_topic in PROMPTS.items():
        target = PER_CAT_TARGETS.get(category, target)
        existing = []
        try:
            entries = bank._read_file(category)
            existing = entries
        except FileNotFoundError:
            pass
        seen = set(existing)

        if len(existing) >= target:
            print(f"[expand-str] {category} 已有 {len(existing)} 条 ≥ {target}，跳过")
            continue

        print(f"[expand-str] {category}: 起点 {len(existing)} → 目标 {target}")
        attempts = 0
        max_attempts = 30
        while len(seen) < target and attempts < max_attempts:
            attempts += 1
            need = target - len(seen)
            ask = min(60, max(30, need))     # foods/passwords 类长 entity 时 ask=100 也撑爆,统一降到 60
            sys_p = (
                "你是一个数据采样器。每次请求返回严格 JSON，形如："
                '{"items": ["item1", "item2", ...]}。'
                "不要添加任何额外说明、Markdown、解释。每个 item 是单独的字符串，不要嵌套结构。"
                "不要重复用户已有的项；不要返回带数字编号或前缀的项（直接返回原始字符串即可）。"
            )
            # avoid 随机抽样 + seed 扰动:每次 attempt 看到不同子集,降低重复率
            # （只展示前 50 个时 LLM 看不到剩下 90%+ 的已有项,生成大量重复浪费配额）
            import random as _r
            _avoid_rng = _r.Random(attempts * 1000 + len(seen))
            avoid_sample = _avoid_rng.sample(list(seen), min(60, len(seen))) if seen else []    # 150 太长,deepseek-flash 卡死
            user_p = (
                f"请输出 {ask} 个【{category}】类别的实体,约束:{prompt_topic}\n\n"
                f"已有 {len(seen)} 条(避开下面随机抽的 {len(avoid_sample)} 个,以及任何与之重复的):\n"
                f"{', '.join(avoid_sample)}\n\n"
                f"严格输出 JSON:{{\"items\": [...]}},items 至少 {ask} 个,优先输出新颖、罕见、未列出的。"
            )
            try:
                resp = call_with_retry(sys_p, user_p, model=model, temperature=0.9, max_tokens=4000)
            except ApiError as e:
                print(f"  [{category}] API 错: {e}")
                break
            try:
                content = resp["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    raise ValueError("content 为空（rate limit / 空响应）")
                obj = json.loads(content)
                items = obj.get("items") or obj.get("data") or []
            except Exception as e:
                # 同时打 raw content 前 200 字便于排查 (JSON 截断、Markdown 包裹、半个 JSON 等)
                raw = (resp.get("choices",[{}])[0].get("message",{}).get("content","") or "")[:200] if 'resp' in dir() else "<no resp>"
                finish = resp.get("choices",[{}])[0].get("finish_reason","?") if 'resp' in dir() else "?"
                print(f"  [{category}] 解析失败: {type(e).__name__}: {str(e)[:120]} | finish={finish} | raw[:200]={raw!r}（sleep 5s 重试）", flush=True)
                import time as _t; _t.sleep(5)
                continue
            added = 0
            for it in items:
                s = str(it).strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                added += 1
            print(f"  [{category}] +{added} → {len(seen)}/{target}（attempt {attempts}）", flush=True)
            # 流式落盘：每次 attempt 后立即写盘，stop/重启不丢进度
            if added > 0:
                write_dict(category, sorted(seen))
            if added == 0:
                break

        # 兜底再写一次
        write_dict(category, sorted(seen))
        print(f"[expand-str] {category} 落盘 {len(seen)} 条", flush=True)


def _expand_corpora(target_pairs: int, model: str) -> None:
    """对话语料扩充。"""
    call_with_retry, ApiError = _resolve_sampler(model)

    PAIR_PROMPTS = {
        "distract_chat": (
            "中文日常闲聊。每个 pair 是一个开放性闲聊话题（user）"
            "+ 自然贴近的回应（assistant，约 15-40 字），话题分散覆盖天气、生活、工作、爱好、"
            "日常琐事、心情等。不要涉及具体的人名、地名、事实数据，保持通用。"
        ),
        "world_qa": (
            "世界知识 QA。每个 pair 是一个具体的事实/常识问题（user）"
            "+ 准确简洁的回答（assistant，约 20-60 字）。覆盖科学、地理、历史、文化、生物、"
            "数学、物理、化学、技术、艺术、体育等主题。要求事实正确，回答精准。"
        ),
    }

    # per-name target 见模块顶部 PER_NAME_TARGETS

    for name, prompt_topic in PAIR_PROMPTS.items():
        existing: list[dict] = []
        try:
            existing = bank._read_pairs_file(name)
        except FileNotFoundError:
            pass
        seen_users = {p["user"] for p in existing}
        all_pairs = list(existing)
        target = PER_NAME_TARGETS.get(name, target_pairs)

        if len(all_pairs) >= target:
            print(f"[expand-pair] {name} 已有 {len(all_pairs)} 条 ≥ {target}，跳过")
            continue

        print(f"[expand-pair] {name}: 起点 {len(all_pairs)} → 目标 {target}")
        attempts = 0
        max_attempts = 60
        while len(all_pairs) < target and attempts < max_attempts:
            attempts += 1
            need = target - len(all_pairs)
            ask = min(80, max(30, need))
            sys_p = (
                "你是一个对话语料生成器。每次请求返回严格 JSON，形如："
                '{"pairs": [{"user": "...", "assistant": "..."}, ...]}。'
                "不要添加任何额外说明、Markdown、解释。每个 pair 必须有 user 和 assistant 两个字段。"
                "不要重复用户已有的 user 串；user 之间话题尽量分散。"
            )
            # avoid 随机抽样 + seed 扰动:与字符串扩充同策略
            import random as _r
            _avoid_rng = _r.Random(attempts * 1000 + len(seen_users))
            avoid_sample = _avoid_rng.sample(list(seen_users), min(80, len(seen_users))) if seen_users else []
            user_p = (
                f"请输出 {ask} 个【{name}】类别的对话 pair,约束:{prompt_topic}\n\n"
                f"已有 {len(seen_users)} 条 user(避开下面随机抽的 {len(avoid_sample)} 个,以及任何与之话题重叠的):\n"
                f"{' / '.join(avoid_sample)}\n\n"
                f"严格输出 JSON:{{\"pairs\": [...]}},pairs 至少 {ask} 个,优先输出新颖、未列出的话题。"
            )
            try:
                resp = call_with_retry(sys_p, user_p, model=model, temperature=0.9, max_tokens=6000)
            except ApiError as e:
                print(f"  [{name}] API 错: {type(e).__name__}: {str(e)[:200]}", flush=True)
                break
            try:
                content = resp["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    raise ValueError("content 为空（rate limit / 空响应）")
                obj = json.loads(content)
                items = obj.get("pairs") or obj.get("data") or []
            except Exception as e:
                raw = (resp.get("choices",[{}])[0].get("message",{}).get("content","") or "")[:200] if 'resp' in dir() else "<no resp>"
                finish = resp.get("choices",[{}])[0].get("finish_reason","?") if 'resp' in dir() else "?"
                print(f"  [{name}] 解析失败: {type(e).__name__}: {str(e)[:120]} | finish={finish} | raw[:200]={raw!r}（sleep 5s 重试）", flush=True)
                import time as _t; _t.sleep(5)
                continue
            added = 0
            for it in items:
                if not isinstance(it, dict):
                    continue
                user = (it.get("user") or "").strip()
                asst = (it.get("assistant") or "").strip()
                if not (user and asst):
                    continue
                if user in seen_users:
                    continue
                seen_users.add(user)
                all_pairs.append({"user": user, "assistant": asst})
                added += 1
            print(f"  [{name}] +{added} → {len(all_pairs)}/{target}（attempt {attempts}）", flush=True)
            # 流式落盘：每次 attempt 后立即写盘
            if added > 0:
                write_pairs(name, all_pairs)
            if added == 0:
                break

        write_pairs(name, all_pairs)
        print(f"[expand-pair] {name} 落盘 {len(all_pairs)} 条", flush=True)


def cmd_expand(target: int = 1000, target_pairs: int = 4500, model: str | None = None) -> None:
    """用 DeepSeek 扩充字符串字典 + 对话语料。"""
    if model is None:
        model = "deepseek-v4-flash"

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    _expand_strings(target=target, model=model)
    _expand_corpora(target_pairs=target_pairs, model=model)

    manifest = write_version_manifest()
    print(f"[expand] dict_version={manifest['dict_version']}", flush=True)


def cmd_version() -> None:
    if not FILES_DIR.exists():
        print(f"[version] 词典目录不存在: {FILES_DIR}")
        sys.exit(1)
    manifest = write_version_manifest()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def cmd_show() -> None:
    cats = bank.list_categories()
    corpora = bank.list_corpora()
    if not cats and not corpora:
        print(f"未检测到素材文件，请先运行 --seed 或 --expand")
        return
    if cats:
        print("[字符串类]")
        for c in cats:
            try:
                train = load_bank(c, "train")
                val = load_bank(c, "val")
                test = load_bank(c, "test")
                print(f"  {c}: train={len(train)}, val={len(val)}, test={len(test)}")
            except Exception as e:
                print(f"  {c}: ERROR {e}")
    if corpora:
        print("[语料类]")
        for c in corpora:
            try:
                train = load_pairs(c, "train")
                val = load_pairs(c, "val")
                test = load_pairs(c, "test")
                print(f"  {c}: train={len(train)}, val={len(val)}, test={len(test)}")
            except Exception as e:
                print(f"  {c}: ERROR {e}")
    print(f"dict_version={bank.dict_version()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", action="store_true", help="用内置最小池子写所有素材")
    g.add_argument("--expand", action="store_true", help="用 DeepSeek 扩充到 target")
    g.add_argument("--version", action="store_true", help="重写 version.json")
    g.add_argument("--show", action="store_true", help="查看当前素材统计")

    ap.add_argument("--target", type=int, default=1000, help="--expand 模式字符串类目标条数")
    ap.add_argument("--target-pairs", type=int, default=5000, help="--expand 模式语料类目标 pair 数")
    ap.add_argument("--model", default=None, help="DeepSeek 模型名（默认 deepseek-v4-flash）")
    args = ap.parse_args()

    if args.seed:
        cmd_seed()
    elif args.expand:
        cmd_expand(target=args.target, target_pairs=args.target_pairs, model=args.model)
    elif args.version:
        cmd_version()
    elif args.show:
        cmd_show()


if __name__ == "__main__":
    main()
