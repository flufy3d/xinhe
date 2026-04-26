"""Beat 3 纯洁性测试。"""
import pytest

from xinhe.data.validator.beat3_purity import check_beat3_purity


def test_clean_beat3_passes():
    """Beat 3 不含 banned，应 OK。"""
    beat3_texts = [
        "今天天气真不错，挺适合出门。",
        "周末打算去公园走走。",
    ]
    banned = ["苹果", "雾蓝色"]
    res = check_beat3_purity(beat3_texts, banned)
    assert res.ok is True


def test_leaked_canonical_fails():
    """Beat 3 含 canonical → 失败。"""
    beat3_texts = ["顺便提一下你之前说的苹果，挺好吃的。"]
    banned = ["苹果"]
    res = check_beat3_purity(beat3_texts, banned)
    assert res.ok is False
    assert "苹果" in res.leaked_terms


def test_leaked_via_fold():
    """Beat 3 通过加空格规避 → 折叠后仍命中。"""
    beat3_texts = ["这个苹 果味道不错。"]
    banned = ["苹果"]
    res = check_beat3_purity(beat3_texts, banned, fold_normalize=True)
    assert res.ok is False


def test_alias_leaked():
    """Beat 3 含 alias → 失败（banned 包含 alias）。"""
    beat3_texts = ["纽约的天气怎么样？"]
    banned = ["纽约", "NYC"]   # canonical + alias 都在 banned 里
    res = check_beat3_purity(beat3_texts, banned)
    assert res.ok is False
    assert "纽约" in res.leaked_terms
