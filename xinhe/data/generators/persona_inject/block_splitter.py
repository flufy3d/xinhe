"""段落 → token 块切分(章内不跨,定长截断)。

输入: NovelIndex.paragraphs + chapter_id_of(由 novel_loader.load_paragraphs 产出)
输出: list[Block],每块 block_size 个 token,块顺序保留(P_n 与 P_{n+1} 在同章相邻)。

切分策略:
  - 按章节分组,每章把段落用 \n 拼起来一次性 tokenize → 章内 token 流
  - 章内 token 流按 block_size 步长非重叠切分,尾段不足 block_size 直接丢弃
  - 章节切换处块序号断开:训练时 (P_n, P_{n+1}) 必须保证 chapter_id 一致

不引入 padding/mask:截断而非 pad,后续 hidden_extractor 直接 [:, -1, :] 拿末 token。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    block_id: int           # 全局编号(0..N-1)
    chapter_id: int         # 所属章节(-1 = 序章前)
    in_chapter_idx: int     # 章内第几块(0..M_chapter-1),用于配对 P_n/P_{n+1}
    token_ids: list[int]
    text_preview: str       # "<头50字> … <尾50字>",debug 用,\n 显示为 ⏎


def _make_preview(chunk: list[int], tokenizer, *, head_chars: int = 50, tail_chars: int = 50) -> str:
    """块预览 = 头 head_chars 字 … 尾 tail_chars 字,\\n 显示为 ⏎(让段落边界可见)。

    定长块按 token count 硬切,块末常落在词中(协议允许:hidden 已吸收前文累积)。
    head+tail 形式让 debug 时容易看出块覆盖范围与切分位置。
    """
    head_raw = tokenizer.decode(chunk[: max(30, head_chars)], skip_special_tokens=True)
    tail_raw = tokenizer.decode(chunk[-max(30, tail_chars):], skip_special_tokens=True)
    head = head_raw.replace("\n", "⏎")[:head_chars]
    tail = tail_raw.replace("\n", "⏎")[-tail_chars:]
    return f"{head} … {tail}"


def split_to_blocks(
    paragraphs: list[str],
    chapter_id_of: list[int],
    tokenizer,
    *,
    block_size: int = 192,
    chapter_aware: bool = True,
) -> list[Block]:
    """章内拼段 + 定长切块。

    Args:
        paragraphs: 段落列表(已剔除章节标题段)。
        chapter_id_of: 同长 list,每段所属章节号。
        tokenizer: HuggingFace tokenizer,需支持 batch encode,不加 special tokens。
        block_size: 块 token 数,默认 192(协议建议 128~256)。
        chapter_aware: True 章末截断,False 全文连续切分(忽略章节边界)。

    Returns:
        list[Block],按章节分组,章内顺序保留。
    """
    if len(paragraphs) != len(chapter_id_of):
        raise ValueError(
            f"paragraphs ({len(paragraphs)}) 与 chapter_id_of ({len(chapter_id_of)}) 长度不一致"
        )

    # 按章节分组(保持原章节顺序)
    if chapter_aware:
        groups: list[tuple[int, list[str]]] = []
        cur_cid = None
        cur_paras: list[str] = []
        for p, cid in zip(paragraphs, chapter_id_of):
            if cid != cur_cid:
                if cur_paras:
                    groups.append((cur_cid, cur_paras))
                cur_cid = cid
                cur_paras = []
            cur_paras.append(p)
        if cur_paras:
            groups.append((cur_cid, cur_paras))
    else:
        groups = [(-99, list(paragraphs))]   # -99 表示忽略章节,所有段同组

    blocks: list[Block] = []
    block_id = 0
    for cid, paras in groups:
        # 一次性 tokenize 整章(\n 连接段落)。不加 special tokens,纯文本流。
        chapter_text = "\n".join(paras)
        token_ids = tokenizer(chapter_text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < block_size:
            continue   # 章太短,丢弃(无法构造一对 P_n/P_{n+1})
        n_blocks = len(token_ids) // block_size
        for i in range(n_blocks):
            start = i * block_size
            chunk = token_ids[start : start + block_size]
            blocks.append(Block(
                block_id=block_id,
                chapter_id=cid,
                in_chapter_idx=i,
                token_ids=chunk,
                text_preview=_make_preview(chunk, tokenizer),
            ))
            block_id += 1

    return blocks


def build_pair_indices(blocks: list[Block]) -> list[tuple[int, int]]:
    """从 blocks 构造合法 (n, n+1) 训练对的全局 idx 列表(同章相邻)。

    在 generator 写盘后,train_persona 也要这个;放在 splitter 里方便复用。
    """
    pairs: list[tuple[int, int]] = []
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        # 同章 + 章内连号 → 合法对
        if a.chapter_id == b.chapter_id and b.in_chapter_idx == a.in_chapter_idx + 1:
            pairs.append((i, i + 1))
    return pairs
