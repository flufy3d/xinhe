"""复现 v11 mix_dynamic 第一批 ep,打印 token 数 / turn 数 / 异常值。"""
import sys, json, random
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    seed = 42
    num_train = 10000
    rng = random.Random(seed)

    items = []
    with open(project_root / "data/skeleton/train.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"loaded {len(items)} ep from train.jsonl")

    rng.shuffle(items)
    taken = items[:num_train]
    print(f"taken first {len(taken)} ep")

    # mix.py 还有第二次 shuffle
    rng.shuffle(taken)
    print(f"after 2nd shuffle, first ep:")

    for i, ep in enumerate(taken[:5]):
        turns = ep.get("conversations", ep.get("turns", []))
        total_chars = sum(len(t.get("content", "")) for t in turns)
        # 估 token 数
        print(f"  ep[{i}] turns={len(turns)} total_chars={total_chars}")
        for j, t in enumerate(turns):
            content = t.get("content", "")
            role = t.get("role", "?")
            print(f"    turn[{j}] role={role} chars={len(content)} : {content[:80]!r}")
        print()


if __name__ == "__main__":
    main()
