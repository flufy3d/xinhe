"""v8 实体词典系统：分类、按 split 隔离、dict_version 哈希。"""
from xinhe.data.dicts.bank import EntityBank, load_bank, list_categories, dict_version

__all__ = ["EntityBank", "load_bank", "list_categories", "dict_version"]
