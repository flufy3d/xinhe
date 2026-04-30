"""
记忆状态机:生成期 / 验证期共享的真值源。

key 三元组 = (subject, relation, scope)
  subject:  user / xiaolin / project_alpha / ...
  relation: favorite_fruit / pet_name / access_code / ...
  scope:    self / third_party / object / ...

mode:
  scalar  —— 单值，新写覆盖旧值（Overwrite 语境）
  set     —— 多值集合，新写追加（Augment 语境）

tombstone:
  True 表示该 key 当前被主动遗忘（L 事件）。再次 WRITE 会清除 tombstone。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional


Key = tuple[str, str, str]


# ── 操作码 ──

OP_WRITE = "WRITE"            # A：单值写入（scalar）
OP_OVERWRITE = "OVERWRITE"    # D：覆盖（"不是 X，是 Y"）
OP_MULTI_WRITE = "MULTI_WRITE"  # F：并发多 key 写入（每 key scalar）
OP_AUGMENT = "AUGMENT"        # J：追加（"也喜欢 Y"）→ set
OP_ERASE = "ERASE"            # L：主动遗忘 → tombstone
OP_ERASE_PARTIAL = "ERASE_PARTIAL"  # L_partial：set 中删指定 value，保留其余
OP_BIND_THIRD = "BIND_THIRD"  # I：第三方实体绑定（声明而非写值）

VALID_OPS = {
    OP_WRITE, OP_OVERWRITE, OP_MULTI_WRITE, OP_AUGMENT,
    OP_ERASE, OP_ERASE_PARTIAL, OP_BIND_THIRD,
}


@dataclass
class FactRecord:
    mode: Literal["scalar", "set"]
    values: list[str]
    active: bool = True
    tombstone: bool = False
    last_op_turn: int = -1


@dataclass
class OpRecord:
    op: str
    old_values: list[str]
    new_values: list[str]
    turn_index: int
    mode: str = "scalar"


class MemoryStateError(ValueError):
    pass


class MemoryState:
    """生成器写入 → 验证器从 conversations 重放 → 比对一致性。

    主要 API:
      apply(op, key, values, turn_index, mode)
      query(key) → 当前 active FactRecord，或 None
      previous_value(key) → 上一个 active values（K/M/L 用）
      is_stale(key) → True 表示 key 上次写入已被覆盖或擦除
      all_active_keys() → 当前仍 active 的 key 列表
    """

    def __init__(self) -> None:
        self.current: dict[Key, FactRecord] = {}
        self.history: dict[Key, list[OpRecord]] = defaultdict(list)

    # ── mutators ──

    def apply(
        self,
        op: str,
        key: Key,
        values: Optional[list[str]] = None,
        turn_index: int = -1,
        mode: str = "scalar",
    ) -> None:
        if op not in VALID_OPS:
            raise MemoryStateError(f"未知 op: {op!r}")

        values = list(values or [])
        prev = self.current.get(key)
        old_values = list(prev.values) if prev else []

        if op == OP_BIND_THIRD:
            # 第三方实体绑定：只声明 subject 存在，不写值
            # 不创建 FactRecord，只记 history（query 该 key 仍返回 None）
            self.history[key].append(OpRecord(op, [], [], turn_index, mode))
            return

        if op in (OP_WRITE, OP_MULTI_WRITE, OP_OVERWRITE):
            new_record = FactRecord(
                mode="scalar",
                values=values,
                active=True,
                tombstone=False,  # 重新写入清除 tombstone
                last_op_turn=turn_index,
            )
            self.current[key] = new_record

        elif op == OP_AUGMENT:
            if prev is None or not prev.active:
                # 没旧值，等价于 WRITE 但 mode=set
                self.current[key] = FactRecord(
                    mode="set", values=values, active=True,
                    tombstone=False, last_op_turn=turn_index,
                )
            else:
                merged = list(prev.values)
                for v in values:
                    if v not in merged:
                        merged.append(v)
                self.current[key] = FactRecord(
                    mode="set", values=merged, active=True,
                    tombstone=False, last_op_turn=turn_index,
                )

        elif op == OP_ERASE:
            self.current[key] = FactRecord(
                mode=prev.mode if prev else "scalar",
                values=[],
                active=False,
                tombstone=True,
                last_op_turn=turn_index,
            )

        elif op == OP_ERASE_PARTIAL:
            if prev is None:
                # 不存在的 key 部分擦除：直接 tombstone
                self.current[key] = FactRecord(
                    mode="set", values=[], active=False,
                    tombstone=True, last_op_turn=turn_index,
                )
            else:
                remaining = [v for v in prev.values if v not in values]
                if remaining:
                    self.current[key] = FactRecord(
                        mode=prev.mode,
                        values=remaining,
                        active=True,
                        tombstone=False,
                        last_op_turn=turn_index,
                    )
                else:
                    self.current[key] = FactRecord(
                        mode=prev.mode,
                        values=[],
                        active=False,
                        tombstone=True,
                        last_op_turn=turn_index,
                    )

        new_values = list(self.current[key].values) if key in self.current else []
        self.history[key].append(
            OpRecord(op, old_values, new_values, turn_index, mode)
        )

    # ── queries ──

    def query(self, key: Key) -> Optional[FactRecord]:
        rec = self.current.get(key)
        if rec is None or not rec.active:
            return None
        return rec

    def previous_value(self, key: Key) -> Optional[list[str]]:
        """返回 key 上一个 active values（即被当前操作覆盖/擦除前的值）。

        - 如果 history 长度 < 2 或上一条 op 后无 active 值，返回 None
        - K/M（Stale-Read / Stale-Miss）和 L 事件用此查询旧值表达
        """
        ops = self.history.get(key, [])
        if len(ops) < 2:
            return None
        # 找最近一条 old_values 非空的 op
        for op in reversed(ops):
            if op.old_values:
                return list(op.old_values)
        return None

    def is_stale(self, key: Key) -> bool:
        rec = self.current.get(key)
        if rec is None:
            return False
        # 至少经历过一次 OVERWRITE / ERASE / ERASE_PARTIAL 才算 stale
        ops = self.history.get(key, [])
        if len(ops) < 2:
            return False
        last = ops[-1]
        return last.op in (OP_OVERWRITE, OP_ERASE, OP_ERASE_PARTIAL)

    def is_tombstoned(self, key: Key) -> bool:
        rec = self.current.get(key)
        return rec is not None and rec.tombstone

    def all_active_keys(self) -> list[Key]:
        return [k for k, r in self.current.items() if r.active]

    def all_keys(self) -> list[Key]:
        return list(self.current.keys())

    # ── 调试 / 序列化 ──

    def snapshot(self) -> dict:
        """导出当前快照（仅供调试与 meta 记录，不入训练数据）。"""
        return {
            "current": {
                f"{k[0]}|{k[1]}|{k[2]}": {
                    "mode": v.mode,
                    "values": v.values,
                    "active": v.active,
                    "tombstone": v.tombstone,
                    "last_op_turn": v.last_op_turn,
                }
                for k, v in self.current.items()
            },
        }
