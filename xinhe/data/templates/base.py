"""
v8 模板基类。

每个原子事件（A..M, L_partial）拥有独立的 templates/{event}.py 文件，
里面定义 ≥8 条 Template，覆盖 8 种 register style。

Template:
  - user_text:  用户侧文本，含占位符 {subject}/{value}/{relation_word}/{ord} 等
  - asst_text:  助手侧文本，含占位符。其中 {value} 必出现在 asst_text 中（用于 char span 抽取）
  - register:   语言风格，便于按风格采样
  - mood:       附加语义（如 "纠错" / "并列"），帮助 D vs J 等近义事件区隔模板池

渲染过程:
  user_rendered = user_text.format(**slots)
  asst_rendered = asst_text.format(**slots)

事件通过 sample_template(pool, rng, register=...) 选模板。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RegisterStyle(str, Enum):
    """8 种语言风格。每事件应至少各覆盖 1 条。"""
    FORMAL = "formal"          # 正式/书面
    CASUAL = "casual"          # 随意/日常
    CLASSICAL = "classical"    # 文白/书卷
    BUSINESS = "business"      # 商务/职场
    GROUP_CHAT = "group_chat"  # 群聊/网络
    TERSE = "terse"            # 短句/省略
    ORAL = "oral"              # 口语/碎碎念
    CUSTOMER = "customer"      # 客服/打断/反问


@dataclass
class Template:
    user_text: str
    asst_text: str
    register: RegisterStyle = RegisterStyle.CASUAL
    mood: str = ""             # 自由标签：纠错 / 并列 / 拒答 / 反问 等
    meta: dict = field(default_factory=dict)  # 关系绑定: relation/bank/scope/mode/soft_eligible

    def render(self, **slots) -> tuple[str, str]:
        """渲染 (user, asst) 文本对。"""
        return self.user_text.format(**slots), self.asst_text.format(**slots)


@dataclass
class TemplatePool:
    """一个事件的全部模板。建议至少 8 条。"""
    event_name: str
    templates: list[Template] = field(default_factory=list)

    def filter(
        self,
        *,
        register: Optional[RegisterStyle] = None,
        mood: Optional[str] = None,
    ) -> list[Template]:
        out = self.templates
        if register is not None:
            out = [t for t in out if t.register == register]
        if mood:
            out = [t for t in out if t.mood == mood]
        return out

    def sample(
        self,
        rng: random.Random,
        *,
        register: Optional[RegisterStyle] = None,
        mood: Optional[str] = None,
    ) -> Template:
        pool = self.filter(register=register, mood=mood)
        if not pool:
            # 若约束过死无候选，回退全集
            pool = self.templates
        if not pool:
            raise ValueError(f"模板池 {self.event_name} 为空")
        return rng.choice(pool)


def sample_template(
    pool: TemplatePool,
    rng: random.Random,
    *,
    register: Optional[RegisterStyle] = None,
    mood: Optional[str] = None,
) -> Template:
    return pool.sample(rng, register=register, mood=mood)
