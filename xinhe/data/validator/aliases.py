"""Alias 表与 Soft 资格判定。

Soft 仅允许：
  - 预定义 alias 精确映射（"NYC" ↔ "纽约"，"小米" ↔ "MI"）
  - 大小写 / 全半角 / 简繁 / 空格 / 标点 折叠后相等

Soft 禁用：
  - 数字、暗号、姓名、地址、颜色细粒度词、否定事实、长度 ≤ 3 的短 value、代码 / ID / 密码 / 序列号
"""
from __future__ import annotations

# 预定义 alias 表：key → 等价别名集合
# 实际使用中，事件生成时记录 canonical 后，validator 用此判定 surface 是否 alias 命中。
ALIAS_MAP: dict[str, list[str]] = {
    # 城市
    "北京": ["首都", "京"],
    "上海": ["沪", "魔都"],
    "广州": ["羊城", "穗"],
    "深圳": ["鹏城"],
    # 品牌
    "Apple": ["苹果", "苹果公司"],
    "Microsoft": ["微软"],
    "Google": ["谷歌"],
    # 其他可按需扩展
}


def get_aliases(canonical: str) -> list[str]:
    return ALIAS_MAP.get(canonical, [])


def is_alias_match(canonical: str, surface: str) -> bool:
    """surface 是否在 canonical 的 alias 列表里。"""
    return surface in ALIAS_MAP.get(canonical, [])


# Soft 禁用的 relation 集合（出自 RelationSpec.soft_eligible=False，校验侧再次硬保证）
SOFT_FORBIDDEN_RELATIONS = {
    # 编号、暗号、姓名、颜色、否定事实
    "project_code", "password", "pet_name", "fav_color",
    "tp_pet_name", "tp_fav_color",
    "name",  # 通用姓名 relation 占位
}


def soft_eligible_for_value(
    value: str,
    *,
    relation: str | None = None,
) -> bool:
    """判定该 value 是否允许 Soft tier。"""
    # 长度 ≤ 3 的短 value 禁 soft
    if len(value) <= 3:
        return False
    # relation 在禁用集合
    if relation and relation in SOFT_FORBIDDEN_RELATIONS:
        return False
    # 全数字 / ID 类（含 - 或字母+数字混合）禁 soft
    has_digit = any(c.isdigit() for c in value)
    only_alnum_punct = all(c.isalnum() or c in "-_/" for c in value)
    if has_digit and only_alnum_punct and len(value) <= 8:
        return False
    return True
