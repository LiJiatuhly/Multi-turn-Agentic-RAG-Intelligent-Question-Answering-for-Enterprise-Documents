from typing import List
from pydantic import BaseModel, Field, field_validator


class QueryAnalysis(BaseModel):
    """问题改写结果的数据结构。"""

    is_clear: bool = Field(
        default=False,
        description="用户的问题是否清晰、可回答。"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="改写后的、自包含的问题列表。"
    )
    clarification_needed: str = Field(
        default="",
        description="如果问题不清晰，说明还需要用户补充什么。"
    )

    @field_validator("is_clear", mode="before")
    @classmethod
    def _coerce_is_clear(cls, v):
        """容错处理：转换字符串形式的布尔值。"""
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "是", "清晰")
        return bool(v)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v):
        """容错处理：确保返回字符串列表。"""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v if str(x).strip()]

    @field_validator("clarification_needed", mode="before")
    @classmethod
    def _coerce_clarification(cls, v):
        """容错处理：确保返回字符串。"""
        if v is None or isinstance(v, bool):
            return ""
        return str(v)
