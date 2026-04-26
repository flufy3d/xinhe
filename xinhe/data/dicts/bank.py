"""
EntityBank：分类实体词典 / 语料库 + train/val/test 切分 + dict_version 哈希。

文件布局：
    xinhe/data/dicts/files/
        surnames.txt          # 字符串类，一行一条
        given_names.txt       # ...
        cities.txt
        ...
        distract_chat.jsonl   # pair 类，一行一个 {"user": "...", "assistant": "..."}
        world_qa.jsonl        # ...
        version.json          # {"dict_version": "v1.0", "split_seed": ..., "categories": {...}}

切分策略（字符串类与 pair 类共享）：
    按 SHA1(key) % 10 决定桶位 → 0-7=train (80%), 8=val (10%), 9=test (10%)
    字符串类用 entity 自身作 key；pair 类用 user 字段作 key。
    每条 entry 在固定 split 中，整库重新 hash 不会被打乱（删/加项除外）。

dict_version：
    汇总所有 txt + jsonl 文件的 SHA1，作为 manifest 写入 version.json。
    任意文件改动都会触发 version 变化。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ── 路径常量 ──

_PKG_ROOT = Path(__file__).resolve().parent
FILES_DIR = _PKG_ROOT / "files"
VERSION_FILE = FILES_DIR / "version.json"


def _hash_split(text: str, n_buckets: int = 10) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_buckets


def _split_for(text: str) -> str:
    """train (0-7), val (8), test (9)。"""
    b = _hash_split(text)
    if b < 8:
        return "train"
    elif b == 8:
        return "val"
    else:
        return "test"


# ── EntityBank ──

@dataclass
class EntityBank:
    category: str
    split: str
    entries: list[str]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError(
                f"EntityBank({self.category}, {self.split}) 为空 —— 请检查 dicts/files/{self.category}.txt"
            )

    def sample(self, rng: random.Random, n: int = 1, *, unique: bool = True) -> list[str]:
        if n == 1:
            return [rng.choice(self.entries)]
        if unique and n <= len(self.entries):
            return rng.sample(self.entries, n)
        return [rng.choice(self.entries) for _ in range(n)]

    def sample_one(self, rng: random.Random) -> str:
        return rng.choice(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


# ── PairBank（语料库：user/assistant pair）──

@dataclass
class PairBank:
    name: str
    split: str
    pairs: list[dict]   # 每个 pair = {"user": str, "assistant": str}

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError(
                f"PairBank({self.name}, {self.split}) 为空 —— 请检查 dicts/files/{self.name}.jsonl"
            )

    def sample_one(self, rng: random.Random) -> dict:
        return rng.choice(self.pairs)

    def sample(self, rng: random.Random, n: int = 1, *, unique: bool = True) -> list[dict]:
        if n == 1:
            return [rng.choice(self.pairs)]
        if unique and n <= len(self.pairs):
            return rng.sample(self.pairs, n)
        return [rng.choice(self.pairs) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.pairs)


# ── 加载逻辑 ──

_CACHE: dict[tuple[str, str], EntityBank] = {}
_PAIR_CACHE: dict[tuple[str, str], PairBank] = {}


def _file_for(category: str) -> Path:
    return FILES_DIR / f"{category}.txt"


def _pair_file_for(name: str) -> Path:
    return FILES_DIR / f"{name}.jsonl"


def _read_file(category: str) -> list[str]:
    p = _file_for(category)
    if not p.exists():
        raise FileNotFoundError(f"缺词典文件: {p}")
    out = []
    seen = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _read_pairs_file(name: str) -> list[dict]:
    p = _pair_file_for(name)
    if not p.exists():
        raise FileNotFoundError(f"缺语料文件: {p}")
    out: list[dict] = []
    seen_users = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        user = obj.get("user")
        asst = obj.get("assistant")
        if not (user and asst):
            continue
        user = str(user).strip()
        asst = str(asst).strip()
        if user in seen_users:
            continue
        seen_users.add(user)
        out.append({"user": user, "assistant": asst})
    return out


def load_bank(category: str, split: str = "train") -> EntityBank:
    """读取某字符串类别在某 split 下的实体集合（缓存）。"""
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split 必须是 train/val/test/all，得到 {split!r}")
    key = (category, split)
    if key in _CACHE:
        return _CACHE[key]

    all_entries = _read_file(category)
    if split == "all":
        entries = all_entries
    else:
        entries = [e for e in all_entries if _split_for(e) == split]

    bank = EntityBank(category=category, split=split, entries=entries)
    _CACHE[key] = bank
    return bank


def load_pairs(name: str, split: str = "train") -> PairBank:
    """读取某 pair 类语料在某 split 下的集合（缓存）。

    split 由 user 字段的 SHA1 决定，与字符串类共享 _split_for。
    """
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split 必须是 train/val/test/all，得到 {split!r}")
    key = (name, split)
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]

    all_pairs = _read_pairs_file(name)
    if split == "all":
        pairs = all_pairs
    else:
        pairs = [p for p in all_pairs if _split_for(p["user"]) == split]

    bank = PairBank(name=name, split=split, pairs=pairs)
    _PAIR_CACHE[key] = bank
    return bank


def list_categories() -> list[str]:
    """字符串类（.txt）。"""
    if not FILES_DIR.exists():
        return []
    return sorted(p.stem for p in FILES_DIR.glob("*.txt"))


def list_corpora() -> list[str]:
    """pair 类（.jsonl）。"""
    if not FILES_DIR.exists():
        return []
    return sorted(p.stem for p in FILES_DIR.glob("*.jsonl"))


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def dict_version() -> str:
    """读 version.json 里的 dict_version；不存在则按文件哈希计算。"""
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text(encoding="utf-8")).get(
                "dict_version", "unversioned"
            )
        except Exception:
            pass
    # fallback: 临时哈希所有文件（含 txt + jsonl）
    if not FILES_DIR.exists():
        return "empty"
    parts = []
    for p in sorted(list(FILES_DIR.glob("*.txt")) + list(FILES_DIR.glob("*.jsonl"))):
        parts.append(p.name + ":" + _hash_file(p)[:12])
    return "h_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def write_version_manifest(version_label: str = "v1.0", split_seed: int = 20260425) -> dict:
    """汇总所有 txt + jsonl 文件的统计 + 哈希，写 version.json。"""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "dict_version": version_label,
        "split_seed": split_seed,
        "categories": {},
        "corpora": {},
    }
    for p in sorted(FILES_DIR.glob("*.txt")):
        entries = _read_file(p.stem)
        train_n = sum(1 for e in entries if _split_for(e) == "train")
        val_n = sum(1 for e in entries if _split_for(e) == "val")
        test_n = sum(1 for e in entries if _split_for(e) == "test")
        manifest["categories"][p.stem] = {
            "total": len(entries),
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "sha1_12": _hash_file(p)[:12],
        }
    for p in sorted(FILES_DIR.glob("*.jsonl")):
        pairs = _read_pairs_file(p.stem)
        train_n = sum(1 for x in pairs if _split_for(x["user"]) == "train")
        val_n = sum(1 for x in pairs if _split_for(x["user"]) == "val")
        test_n = sum(1 for x in pairs if _split_for(x["user"]) == "test")
        manifest["corpora"][p.stem] = {
            "total": len(pairs),
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "sha1_12": _hash_file(p)[:12],
        }
    VERSION_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ── 写入工具（被 build_dicts.py 调用）──

def write_dict(category: str, entries: Iterable[str], *, dedup: bool = True) -> int:
    """把 entries 合并写到 files/{category}.txt(读现状 + 新增,去重写回,**不覆盖** user 手写新增)。

    防 race condition: driver 在 expand 时写盘 间 user 手编辑文件,
    旧实现 truncate-and-write 会丢用户新增。新实现 read-merge-write。
    """
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    p = _file_for(category)
    # 先读 file 现状
    existing: list[str] = []
    if p.exists():
        try:
            existing = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            pass
    # 合并 file 现状 + 新增 entries(file 优先 → 不破坏 user 手写顺序)
    seen = set()
    out = []
    for source in (existing, entries):
        for e in source:
            s = (e or "").strip()
            if not s:
                continue
            if dedup and s in seen:
                continue
            seen.add(s)
            out.append(s)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    for k in list(_CACHE.keys()):
        if k[0] == category:
            del _CACHE[k]
    return len(out)


def write_pairs(name: str, pairs: Iterable[dict], *, dedup: bool = True) -> int:
    """把 pairs 写到 files/{name}.jsonl（覆盖式写）。

    每个 pair 必须有 user / assistant 字段；按 user 去重。
    """
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    seen_users = set()
    out: list[dict] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        user = (pair.get("user") or "").strip()
        asst = (pair.get("assistant") or "").strip()
        if not (user and asst):
            continue
        if dedup and user in seen_users:
            continue
        seen_users.add(user)
        out.append({"user": user, "assistant": asst})
    p = _pair_file_for(name)
    with open(p, "w", encoding="utf-8") as fp:
        for pair in out:
            fp.write(json.dumps(pair, ensure_ascii=False) + "\n")
    # 失效缓存
    for k in list(_PAIR_CACHE.keys()):
        if k[0] == name:
            del _PAIR_CACHE[k]
    return len(out)
