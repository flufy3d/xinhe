"""normalize 折叠 + tier 分级单元测试。

关键边界：
  - 折叠后 A17 ≠ A71（数字结构保留）
  - 折叠后 蓝色 ≠ 绿色
  - 数字 / 编号 / 姓名 / 颜色 禁用 Soft tier
  - len ≤ 3 的 value 禁 Soft
"""
import pytest

from xinhe.data.validator.normalize import fold, fold_keep_id
from xinhe.data.validator.tier import classify_tier, TierVerdict
from xinhe.data.validator.aliases import soft_eligible_for_value


# ── normalize ──

def test_fold_punctuation():
    assert fold("Hello, World!") == "helloworld"


def test_fold_fullwidth_to_halfwidth():
    assert fold("ＡＢＣ１２３") == "abc123"


def test_fold_preserves_digit_distinction():
    # A17 与 A71 折叠后必须不同
    assert fold("A17") != fold("A71")


def test_fold_preserves_color_distinction():
    # 颜色不能因折叠混淆
    assert fold("蓝色") != fold("绿色")


def test_fold_keep_id_preserves_dash():
    # ID 类折叠保留连字符
    assert "-" in fold_keep_id("K9Q-27")


# ── tier ──

def test_classify_hard_exact_substring():
    res = classify_tier("苹果", "我喜欢吃苹果。")
    assert res.verdict == TierVerdict.HARD
    assert res.span is not None


def test_classify_hard_through_fullwidth_fold():
    # value 是 "123" 但 content 是 "１２３"（全角），折叠后命中
    res = classify_tier("123", "编号是１２３。")
    assert res.verdict == TierVerdict.HARD


def test_classify_reject_when_absent():
    res = classify_tier("苹果", "我喜欢吃梨。")
    assert res.verdict == TierVerdict.REJECT


def test_classify_reject_visually_similar():
    # A17 不能匹配到 A71（编号语义敏感）
    res = classify_tier("A17", "编号是 A71。")
    assert res.verdict == TierVerdict.REJECT


# ── soft eligibility ──

def test_soft_forbidden_short_values():
    # len ≤ 3 禁 soft
    assert soft_eligible_for_value("AB") is False
    assert soft_eligible_for_value("123") is False


def test_soft_forbidden_for_password_relation():
    assert soft_eligible_for_value("MyPassword", relation="password") is False


def test_soft_forbidden_for_pet_name_relation():
    assert soft_eligible_for_value("Leo", relation="pet_name") is False


def test_soft_forbidden_for_color_relation():
    assert soft_eligible_for_value("蓝色", relation="fav_color") is False


def test_soft_forbidden_for_id_pattern():
    # 字母数字混合短串（K9Q）禁 soft
    assert soft_eligible_for_value("K9Q-27", relation="project_code") is False


def test_soft_eligible_for_long_food():
    # 长 value + 食物 relation → 允许 soft
    assert soft_eligible_for_value("红烧排骨", relation="fav_food") is True
