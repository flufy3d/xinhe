"""验证器:四元一致性 + memory_state 重放 + Hard/Soft/Reject + Beat 3 纯洁性。"""
from xinhe.data.validator.api import (
    ValidationResult,
    validate,
)
from xinhe.data.validator.normalize import fold
from xinhe.data.validator.tier import classify_tier, TierVerdict

__all__ = [
    "ValidationResult",
    "validate",
    "fold",
    "classify_tier",
    "TierVerdict",
]
