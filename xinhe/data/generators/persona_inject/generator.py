"""PersonaInjectGenerator: 整合 splitter + extractor → 写盘。

不入 GENERATORS dispatcher,因为产出是张量(.pt)而非 jsonl。
由 scripts/generate_persona_data.py 直接调。

输出目录布局:
    <out_dir>/<novel_stem>/
    ├── manifest.json        # 元数据 + 缓存指纹(novel_sha256, layer_indices)
    ├── blocks.jsonl         # 每行 {block_id, chapter_id, in_chapter_idx, n_tokens, text_preview}
    ├── pairs.json           # 合法 (n, n+1) 对的 idx 列表(同章相邻)
    └── hidden_states.pt     # bf16 Tensor[N_blocks, len(layer_indices), hidden_dim]
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from xinhe.data.generators.novel.novel_loader import (
    DEFAULT_CHAPTER_PATTERN,
    load_paragraphs,
)
from xinhe.data.generators.persona_inject.block_splitter import (
    Block,
    build_pair_indices,
    split_to_blocks,
)
from xinhe.data.generators.persona_inject.hidden_extractor import (
    extract_hidden_states,
    get_backbone_dims,
)


def _file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_out_name(raw: str, *, fallback: str = "") -> str:
    """把任意字符串清成跨平台安全的目录名(ASCII letters/digits/._-)。

    多个非法字符压缩成一个 _,头尾的 _ 删掉。结果为空时 → 用 fallback。
    """
    s = _SAFE_NAME_RE.sub("_", raw).strip("_.-")
    return s or fallback


def _default_out_name(novel_path: Path) -> str:
    """从 novel_path.stem 推默认目录名:sanitize 后若空 → sha256[:12]。"""
    cleaned = sanitize_out_name(novel_path.stem)
    if cleaned:
        return cleaned
    # 中文/纯特殊字符 stem 完全被清空 → fallback 到稳定哈希前缀
    return "novel_" + _file_sha256(novel_path)[:12]


class PersonaInjectGenerator:
    name = "persona_inject"

    def __init__(
        self,
        *,
        novel_path: str,
        out_dir: str,
        backbone_model_path: str,
        layer_indices: Sequence[int],
        out_name: str | None = None,    # 子目录名;None → 自动 sanitize stem,空则 sha 前缀
        block_size: int = 192,
        chapter_aware: bool = True,
        extract_batch_size: int = 8,
        extract_dtype: str = "bfloat16",
        device: str = "cuda",
        chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
        min_dialog_density: float = 0.0,   # persona 用整本小说,不强求对话密度
    ):
        self.novel_path = Path(novel_path)
        # 用 sanitize 过的名字做子目录,避免 Linux 下中文/空格/特殊字符路径问题
        self.out_name = (
            sanitize_out_name(out_name, fallback="") if out_name
            else _default_out_name(self.novel_path)
        )
        if not self.out_name:
            raise ValueError(f"out_name 经 sanitize 后为空: {out_name!r}")
        self.out_dir = Path(out_dir) / self.out_name
        self.backbone_model_path = backbone_model_path
        self.layer_indices = list(layer_indices)
        self.block_size = block_size
        self.chapter_aware = chapter_aware
        self.extract_batch_size = extract_batch_size
        self.extract_dtype_str = extract_dtype
        self.device = device
        self.chapter_pattern = chapter_pattern
        self.min_dialog_density = min_dialog_density

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def blocks_path(self) -> Path:
        return self.out_dir / "blocks.jsonl"

    @property
    def pairs_path(self) -> Path:
        return self.out_dir / "pairs.json"

    @property
    def hidden_path(self) -> Path:
        return self.out_dir / "hidden_states.pt"

    def _is_cached(self, novel_sha: str) -> bool:
        """只有完整四件套都在 + 指纹一致(同书 + 同层 + 同 block_size)→ 命中。"""
        if not self.manifest_path.exists():
            return False
        try:
            mf = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if mf.get("novel_sha256") != novel_sha:
            return False
        if mf.get("layer_indices") != self.layer_indices:
            return False
        if mf.get("block_size") != self.block_size:
            return False
        for p in (self.blocks_path, self.pairs_path, self.hidden_path):
            if not p.exists():
                return False
        return True

    def generate(self, *, force: bool = False) -> dict:
        if not self.novel_path.exists():
            raise FileNotFoundError(f"小说文件不存在: {self.novel_path}")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        novel_sha = _file_sha256(self.novel_path)

        if not force and self._is_cached(novel_sha):
            mf = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            print(f"  [persona_inject] 缓存命中 → {self.out_dir} ({mf.get('n_blocks')} 块)")
            return mf

        # 1) backbone 形参校验(不加载权重)
        n_layers, hidden_size, intermediate_size = get_backbone_dims(self.backbone_model_path)
        if max(self.layer_indices) >= n_layers or min(self.layer_indices) < 0:
            raise ValueError(
                f"layer_indices {self.layer_indices} 越界: backbone {n_layers} 层"
            )
        print(
            f"  [persona_inject] backbone {n_layers} 层 / hidden={hidden_size} / "
            f"intermediate={intermediate_size}"
        )

        # 2) tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.backbone_model_path, trust_remote_code=True
        )

        # 3) 段落
        idx = load_paragraphs(
            self.novel_path,
            min_density=self.min_dialog_density,
            chapter_pattern=self.chapter_pattern,
        )
        print(
            f"  [persona_inject] {self.novel_path.name}: "
            f"{idx.n_paragraphs} 段, {idx.n_chapters} 章, "
            f"dialog 密度 {idx.dialog_density:.1%}"
        )

        # 4) 切块
        blocks = split_to_blocks(
            idx.paragraphs, idx.chapter_id_of, tokenizer,
            block_size=self.block_size, chapter_aware=self.chapter_aware,
        )
        if not blocks:
            raise RuntimeError(
                f"切块后为空: 章太短 (< {self.block_size} tokens)? 试 block_size 减半"
            )
        pairs = build_pair_indices(blocks)
        print(f"  [persona_inject] 切块: {len(blocks)} 块, {len(pairs)} 个合法 (n,n+1) 对")
        if not pairs:
            raise RuntimeError(
                "无合法训练对(每章 < 2 块)。增大输入或减小 block_size。"
            )

        # 5) 提取
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(self.extract_dtype_str)
        if dtype is None:
            raise ValueError(f"未知 extract_dtype: {self.extract_dtype_str!r}")
        hidden = extract_hidden_states(
            blocks,
            backbone_model_path=self.backbone_model_path,
            layer_indices=self.layer_indices,
            batch_size=self.extract_batch_size,
            device=self.device,
            dtype=dtype,
        )

        # 6) 写盘
        torch.save(hidden, self.hidden_path)
        with open(self.blocks_path, "w", encoding="utf-8") as f:
            for b in blocks:
                f.write(json.dumps({
                    "block_id": b.block_id,
                    "chapter_id": b.chapter_id,
                    "in_chapter_idx": b.in_chapter_idx,
                    "n_tokens": len(b.token_ids),
                    "text_preview": b.text_preview,
                }, ensure_ascii=False) + "\n")
        self.pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

        manifest = {
            "novel_path": str(self.novel_path),
            "novel_filename": self.novel_path.name,
            "novel_sha256": novel_sha,
            "novel_stem": self.novel_path.stem,    # 原始 stem(可能含中文/空格)
            "out_name": self.out_name,             # 实际目录名(sanitize 后)
            "backbone_model_path": self.backbone_model_path,
            "tokenizer": Path(self.backbone_model_path).name,
            "n_layers_backbone": n_layers,
            "hidden_size": hidden_size,
            "intermediate_size_backbone": intermediate_size,
            "layer_indices": self.layer_indices,
            "n_layers_kept": len(self.layer_indices),
            "block_size": self.block_size,
            "chapter_aware": self.chapter_aware,
            "n_blocks": len(blocks),
            "n_pairs": len(pairs),
            "n_chapters_used": len({b.chapter_id for b in blocks}),
            "extract_dtype": self.extract_dtype_str,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"  [persona_inject] 写盘完成 → {self.out_dir}")
        return manifest
