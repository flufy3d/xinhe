"""
构建 DeepSeek teacher cache — general_chat 和 world_qa turn 池。

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    python scripts/build_chat_cache.py --n-chat 30000 --n-qa 20000

推荐 off-peak 跑（北京时间 00:30-08:30，DeepSeek 半价）。

输出:
    data/cache/general_chat.jsonl  — 每行 {"type": "general_chat", "user": ..., "assistant": ...}
    data/cache/world_qa.jsonl       — 每行 {"type": "world_qa", "user": ..., "assistant": ...}

特性:
    - 断点续传：重启时读取已有 jsonl 计数，跳过已完成部分
    - 每次 API call 生成 N 个 turn，均摊开销
    - 质量过滤：长度 / 重复 / 中文 / refusal 冲突
    - key 永不落盘，只从 DEEPSEEK_API_KEY 环境变量读
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.deepseek_sampler import (
    DeepSeekError, call_with_retry, extract_turns, quality_filter,
)


# ─── General chat 的 seed 话题 ───
# 日常、情绪、观点、感受类，不考事实
CHAT_SEED_TOPICS = [
    "今天天气", "最近忙不忙", "心情", "下班后干啥", "周末计划",
    "工作压力", "失眠", "吃饭", "喝什么", "咖啡还是茶",
    "早起困难", "运动", "锻炼", "散步", "宅家",
    "看什么剧", "电影", "音乐", "追番", "游戏",
    "朋友圈", "社交", "独处", "旅行向往", "城市生活",
    "乡愁", "家乡", "天气变化", "换季", "过年",
    "长辈催婚", "同事相处", "领导", "加班", "996",
    "裸辞", "换工作", "升职", "做副业", "学新东西",
    "看书", "推荐书", "Kindle", "微信读书", "听播客",
    "博客", "公众号", "B站", "知乎", "小红书",
    "买东西", "快递", "外卖", "网购", "拼多多",
    "理财", "存钱", "消费观", "买房", "租房",
    "房租涨价", "城市房价", "养老", "保险", "医保",
    "生病", "感冒", "头疼", "失眠多梦", "养生",
    "中医", "拔罐", "艾灸", "食疗", "早睡",
    "咖啡戒断", "奶茶上瘾", "减肥", "健身", "瑜伽",
    "跑步", "跳绳", "骑行", "登山", "骑行装备",
    "相机", "摄影爱好", "手机摄影", "剪辑", "vlog",
    "家具", "极简", "断舍离", "收纳", "装修",
    "养花", "种菜", "绿植", "多肉", "花盆",
    "AI工具", "ChatGPT", "AI画图", "AI写作", "未来",
    "时间管理", "GTD", "拖延", "习惯养成", "早睡计划",
    "冥想", "正念", "心流", "心理学", "原生家庭",
    "异地恋", "分手", "相亲", "单身", "朋友介绍",
    "宠物", "养猫", "养狗", "铲屎官", "遛狗",
    "陌生人", "打招呼", "电梯尴尬", "社恐", "社牛",
    "KTV", "聚餐", "团建", "同学聚会", "老朋友",
    "过年回家", "春运", "车票", "机票", "旅途",
    "飞机颠簸", "高铁", "地铁", "公交", "打车",
    "天气预报", "下雨", "雾霾", "台风", "春天",
    "夏天", "秋天", "冬天", "降温", "开空调",
]

# ─── World QA 的 seed 话题 ───
# 考事实的，可复查的
QA_SEED_TOPICS = [
    "中国地理", "世界地理", "首都", "大洲", "海洋",
    "山脉", "河流", "沙漠", "高原", "盆地",
    "历史朝代", "唐朝", "宋朝", "明朝", "清朝",
    "古代人物", "诗人", "科学家", "发明家", "政治家",
    "二战", "一战", "大航海", "文艺复兴", "工业革命",
    "物理常识", "牛顿定律", "万有引力", "光速", "相对论",
    "化学", "元素周期表", "水分子", "酸碱", "金属",
    "生物", "细胞", "DNA", "光合作用", "进化",
    "天文", "太阳系", "行星", "黑洞", "宇宙",
    "数学", "圆周率", "质数", "勾股定理", "函数",
    "动物", "哺乳动物", "鸟类", "爬行", "恐龙",
    "植物", "树", "花", "果", "农作物",
    "食物", "营养", "维生素", "蛋白质", "碳水",
    "医学常识", "心脏", "大脑", "血液", "免疫",
    "计算机", "互联网", "编程语言", "CPU", "操作系统",
    "AI基础", "神经网络", "深度学习", "机器学习", "Transformer",
    "经济学", "GDP", "通货膨胀", "股票", "汇率",
    "法律常识", "宪法", "民法", "刑法", "合同",
    "文学", "四大名著", "小说", "现代文学", "外国文学",
    "艺术", "绘画", "雕塑", "音乐", "舞蹈",
    "建筑", "故宫", "长城", "金字塔", "埃菲尔铁塔",
    "体育", "奥运会", "足球", "篮球", "乒乓球",
    "哲学", "儒家", "道家", "西方哲学", "存在主义",
    "宗教常识", "佛教", "基督教", "伊斯兰教", "道教",
    "节气", "二十四节气", "立春", "夏至", "冬至",
    "节日", "春节", "中秋", "端午", "国庆",
    "菜系", "川菜", "粤菜", "鲁菜", "淮扬菜",
    "名人", "鲁迅", "孔子", "爱因斯坦", "达芬奇",
    "国家", "美国", "日本", "英国", "德国",
    "货币", "人民币", "美元", "欧元", "日元",
]


CHAT_SYSTEM_PROMPT = """你是一个中文对话数据生成助手。请生成一组 {n_turns} 轮自然的中文聊天对话。

要求:
1. 每轮是 user + assistant 的一问一答
2. 用户提出日常话题（围绕给定主题），assistant 给出自然、简洁（1-3 句）的回复
3. 回复要有人情味，表达情绪/观点/共鸣，不要主动追问个人信息
4. 不要涉及具体事实答案（这组用于闲聊场景，不是问答）
5. assistant 回复**不要**包含"还没告诉我"、"不知道"、"没提过"之类的拒答语

输出严格的 JSON 格式:
{{
  "turns": [
    {{"user": "...", "assistant": "..."}},
    ...
  ]
}}
"""

QA_SYSTEM_PROMPT = """你是一个中文事实问答数据生成助手。请生成一组 {n_turns} 条中文常识问答。

要求:
1. 每条是 user 提问 + assistant 回答，围绕给定主题的世界知识
2. 问题要清晰可查，答案要正确、简洁（1-2 句），不要"大概"、"可能"等不确定表述
3. 答案里应有明确的事实要点（如地名、人名、年代、数字、概念）
4. 问题和答案都用中文
5. 每条 Q/A 独立，不要前后文关联

输出严格的 JSON 格式:
{{
  "turns": [
    {{"user": "...", "assistant": "..."}},
    ...
  ]
}}
"""


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def sample_batch(category: str, seed_topic: str, n_turns: int, model: str) -> list[dict]:
    """采一个 batch。category: 'general_chat' or 'world_qa'。"""
    if category == "general_chat":
        sys_prompt = CHAT_SYSTEM_PROMPT.format(n_turns=n_turns)
        user_prompt = f"话题: {seed_topic}\n\n请按要求输出 JSON。"
        temp = 0.85
    elif category == "world_qa":
        sys_prompt = QA_SYSTEM_PROMPT.format(n_turns=n_turns)
        user_prompt = f"话题: {seed_topic}\n\n请按要求输出 JSON。"
        temp = 0.4   # 事实问答温度低一些，避免幻觉
    else:
        raise ValueError(category)

    resp = call_with_retry(
        sys_prompt, user_prompt, model=model,
        temperature=temp, top_p=0.9, max_tokens=4000,
    )
    turns = extract_turns(resp)
    # 质量过滤 + 标记 type
    out = []
    for t in turns:
        if quality_filter(t, category):
            t["type"] = category
            out.append(t)
    return out


def build(
    output_path: Path,
    category: str,
    target_count: int,
    model: str,
    turns_per_call: int,
    seed_topics: list,
    rng: random.Random,
    min_delay: float,
    concurrency: int = 16,
):
    """增量构建 cache 文件，已有 N 条时从 N 条起继续。

    并发采样：concurrency 路并行 call，主线程持文件锁按完成顺序 append。
    目标达到后不再派新任务（已在途的若回来仍写入，但 current>=target 时跳过 append）。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current = count_lines(output_path)
    print(f"[{category}] 目标 {target_count} 条，已有 {current} 条，并发 {concurrency}")
    if current >= target_count:
        print(f"[{category}] 已满足目标，跳过。")
        return

    # 线程安全计数 + 文件写锁
    counter_lock = threading.Lock()
    write_lock = threading.Lock()
    state = {"current": current, "submitted": 0}
    start_all = time.time()

    def worker(topic: str) -> tuple[str, list]:
        """单次采样，返回 (topic, turns_list)；失败返回空 list。"""
        try:
            t0 = time.time()
            turns = sample_batch(category, topic, turns_per_call, model)
            return (topic, turns, time.time() - t0)
        except DeepSeekError as e:
            return (topic, [], 0.0)

    f = open(output_path, "a", encoding="utf-8")
    try:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = set()

            # 初始派 concurrency 个任务（不过量预填，避免 target 命中后大量 tail 浪费）
            for _ in range(concurrency):
                topic = rng.choice(seed_topics)
                futures.add(pool.submit(worker, topic))
                state["submitted"] += 1

            while futures:
                # 等任一完成
                done = next(as_completed(futures))
                futures.remove(done)

                try:
                    topic, turns, elapsed = done.result()
                except Exception as e:
                    print(f"  [{category}] worker 异常: {e}", flush=True)
                    turns = []
                    elapsed = 0.0

                # 写入
                if turns:
                    with write_lock:
                        added = 0
                        for t in turns:
                            with counter_lock:
                                if state["current"] >= target_count:
                                    break
                                state["current"] += 1
                                added += 1
                            f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        f.flush()
                    with counter_lock:
                        cur = state["current"]
                    total_elapsed = time.time() - start_all
                    rate = (cur - current) / max(total_elapsed, 0.01)
                    print(f"  [{category}] +{added} ({elapsed:.1f}s/call) | 话题={topic} | 总 {cur}/{target_count} | 平均 {rate:.1f}/s", flush=True)

                # 如果还没到 target，补派任务（保持 concurrency 在途）
                with counter_lock:
                    need_more = state["current"] < target_count
                if need_more:
                    topic = rng.choice(seed_topics)
                    futures.add(pool.submit(worker, topic))
                    state["submitted"] += 1
                else:
                    # 已达标：取消所有未开跑的 future（Python 3.9+ ThreadPoolExecutor 支持）
                    for fut in futures:
                        fut.cancel()
                    # 从 set 里移除已取消的
                    futures = {fut for fut in futures if not fut.cancelled()}
        finally:
            # 不等待剩余在跑的 call（它们已经不贡献任何有用输出）
            pool.shutdown(wait=False, cancel_futures=True)
    finally:
        f.close()


def _estimated_calls_needed(target: int, current: int, turns_per_call: int) -> int:
    """估算最多需要多少 call（考虑过滤失败率 20%）"""
    remaining = max(target - current, 0)
    return max(int(remaining / turns_per_call * 1.5) + 10, 20)


def main():
    p = argparse.ArgumentParser(description="DeepSeek teacher cache 构建")
    p.add_argument("--n-chat", type=int, default=30000, help="general_chat 目标条数")
    p.add_argument("--n-qa", type=int, default=20000, help="world_qa 目标条数")
    p.add_argument("--turns-per-call", type=int, default=30, help="单次 API call 生成 turn 数")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--out-dir", type=str, default="data/cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-delay", type=float, default=1.0, help="call 间最小间隔秒数")
    p.add_argument("--concurrency", type=int, default=16, help="并发 API call 数量")
    p.add_argument("--category", type=str, default="both",
                   choices=["general_chat", "world_qa", "both"],
                   help="只跑哪类")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    chat_path = out_dir / "general_chat.jsonl"
    qa_path = out_dir / "world_qa.jsonl"

    # 预先检测 API key 存在（避免跑到一半才报错）
    from xinhe.data.deepseek_sampler import _load_api_key
    try:
        _load_api_key()
    except DeepSeekError as e:
        print(f"错误: {e}")
        sys.exit(1)
    print(f"[OK] DEEPSEEK_API_KEY 已检测到（未显示）")
    print(f"[model] {args.model}")
    print(f"[min_delay] {args.min_delay}s")

    if args.category in ("general_chat", "both"):
        build(chat_path, "general_chat", args.n_chat, args.model,
              args.turns_per_call, CHAT_SEED_TOPICS, rng, args.min_delay,
              concurrency=args.concurrency)

    if args.category in ("world_qa", "both"):
        build(qa_path, "world_qa", args.n_qa, args.model,
              args.turns_per_call, QA_SEED_TOPICS, rng, args.min_delay,
              concurrency=args.concurrency)

    print("\n[完成]")
    print(f"  general_chat: {count_lines(chat_path)} 条 → {chat_path}")
    print(f"  world_qa:      {count_lines(qa_path)} 条 → {qa_path}")


if __name__ == "__main__":
    main()
