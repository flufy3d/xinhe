"""一次性 inspect skeleton val.jsonl 分布,看 read turn 占比 / 距离分布。"""
import json
import sys
from collections import Counter
from pathlib import Path

val_path = sys.argv[1] if len(sys.argv) > 1 else "data/skeleton/val.jsonl"

skel_counter = Counter()
distances = []  # write→read 距离
recall_counter = Counter()

with open(val_path, encoding="utf-8") as f:
    for line in f:
        ep = json.loads(line)
        skel = ep["skeleton_id"]
        skel_counter[skel] += 1
        convs = ep["conversations"]

        # 找 write/read 配对:同一 value 字符串第一次出现的位置 = write,后续 = read
        seen_value = {}
        for turn_idx, t in enumerate(convs):
            if t["role"] != "assistant":
                continue
            vals = t.get("value") or []
            for v in vals:
                if v not in seen_value:
                    seen_value[v] = turn_idx
                else:
                    write_idx = seen_value[v]
                    dist = (turn_idx - write_idx) // 2  # 每对 user/asst 算一轮
                    distances.append(dist)
                    # is_recall: value 不在 user_msg 里
                    user_msg = convs[turn_idx - 1].get("content", "") if turn_idx > 0 else ""
                    recall_counter[v not in user_msg] += 1

print(f"=== skeleton 分布 ===")
for k, v in sorted(skel_counter.items()):
    print(f"  {k}: {v}")
print(f"\n=== write→read 距离分布 (turns) ===")
dist_counter = Counter(distances)
for d, n in sorted(dist_counter.items()):
    print(f"  {d}: {n}")
print(f"\n=== is_recall (value not in user_msg) ===")
print(f"  True (真测记忆): {recall_counter[True]}")
print(f"  False (写/echo): {recall_counter[False]}")
