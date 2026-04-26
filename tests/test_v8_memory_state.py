"""MemoryState 单元测试：apply 各 op、scalar/set 切换、tombstone、history 重放。"""
import pytest

from xinhe.data.memory_state import (
    MemoryState,
    OP_WRITE,
    OP_OVERWRITE,
    OP_AUGMENT,
    OP_ERASE,
    OP_ERASE_PARTIAL,
    OP_BIND_THIRD,
)


def test_write_then_query():
    s = MemoryState()
    k = ("user", "fav_food", "self")
    s.apply(OP_WRITE, k, ["苹果"], turn_index=0)
    rec = s.query(k)
    assert rec is not None
    assert rec.values == ["苹果"]
    assert rec.active is True


def test_overwrite_replaces_value():
    s = MemoryState()
    k = ("user", "fav_food", "self")
    s.apply(OP_WRITE, k, ["苹果"], 0)
    s.apply(OP_OVERWRITE, k, ["梨"], 1)
    assert s.query(k).values == ["梨"]
    assert s.previous_value(k) == ["苹果"]


def test_augment_adds_to_set():
    s = MemoryState()
    k = ("user", "fav_color", "self")
    s.apply(OP_WRITE, k, ["蓝色"], 0)
    s.apply(OP_AUGMENT, k, ["绿色"], 1)
    rec = s.query(k)
    assert set(rec.values) == {"蓝色", "绿色"}
    assert rec.mode == "set"


def test_erase_creates_tombstone():
    s = MemoryState()
    k = ("user", "pet_name", "self")
    s.apply(OP_WRITE, k, ["Leo"], 0)
    s.apply(OP_ERASE, k, [], 1)
    assert s.query(k) is None
    assert s.is_tombstoned(k) is True
    assert s.previous_value(k) == ["Leo"]


def test_rewrite_clears_tombstone():
    s = MemoryState()
    k = ("user", "pet_name", "self")
    s.apply(OP_WRITE, k, ["Leo"], 0)
    s.apply(OP_ERASE, k, [], 1)
    s.apply(OP_WRITE, k, ["Max"], 2)
    assert s.is_tombstoned(k) is False
    assert s.query(k).values == ["Max"]


def test_partial_erase_keeps_remaining():
    s = MemoryState()
    k = ("user", "fav_color", "self")
    s.apply(OP_WRITE, k, ["蓝色", "绿色", "红色"], 0, mode="set")
    s.apply(OP_ERASE_PARTIAL, k, ["绿色"], 1)
    rec = s.query(k)
    assert rec is not None
    assert "绿色" not in rec.values
    assert "蓝色" in rec.values and "红色" in rec.values


def test_partial_erase_all_tombstones():
    s = MemoryState()
    k = ("user", "fav_color", "self")
    s.apply(OP_WRITE, k, ["蓝色"], 0, mode="set")
    s.apply(OP_ERASE_PARTIAL, k, ["蓝色"], 1)
    assert s.query(k) is None
    assert s.is_tombstoned(k) is True


def test_bind_third_does_not_create_record():
    s = MemoryState()
    k = ("xiaolin", "tp_pet_kind", "third_party")
    s.apply(OP_BIND_THIRD, k, [], 0)
    assert s.query(k) is None
    # 后续 WRITE 才创建
    s.apply(OP_WRITE, k, ["猫"], 1)
    assert s.query(k).values == ["猫"]


def test_history_records_old_values():
    s = MemoryState()
    k = ("user", "fav_food", "self")
    s.apply(OP_WRITE, k, ["苹果"], 0)
    s.apply(OP_OVERWRITE, k, ["梨"], 1)
    s.apply(OP_OVERWRITE, k, ["香蕉"], 2)
    # previous_value 应返回最近一次非空 old_values
    assert s.previous_value(k) == ["梨"]


def test_is_stale_after_overwrite():
    s = MemoryState()
    k = ("user", "fav_food", "self")
    s.apply(OP_WRITE, k, ["苹果"], 0)
    assert s.is_stale(k) is False
    s.apply(OP_OVERWRITE, k, ["梨"], 1)
    assert s.is_stale(k) is True


def test_all_active_keys_excludes_tombstoned():
    s = MemoryState()
    k1 = ("user", "fav_food", "self")
    k2 = ("user", "fav_color", "self")
    s.apply(OP_WRITE, k1, ["苹果"], 0)
    s.apply(OP_WRITE, k2, ["蓝色"], 1)
    s.apply(OP_ERASE, k1, [], 2)
    keys = s.all_active_keys()
    assert k2 in keys
    assert k1 not in keys
