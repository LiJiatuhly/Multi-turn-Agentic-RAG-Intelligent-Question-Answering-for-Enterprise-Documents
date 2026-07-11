# 结构化输出的数据模型：定义 rewrite_query 节点期望模型返回的 JSON 结构。

from typing import List
from pydantic import BaseModel, Field, field_validator


class QueryAnalysis(BaseModel):
    """B 级 问题改写结果的数据结构。

    LangGraph 的 with_structured_output(QueryAnalysis, method="json_mode") 会：
      1. 在系统提示里附上这个结构的 JSON 格式要求
      2. 要求模型以 JSON 格式返回
      3. 把模型返回的 JSON 自动解析成这个类的实例

    为什么要 field_validator（容错校验器）？
    智谱用 json_mode 时，偶尔会把 clarification_needed 返回成布尔值（而不是字符串）。
    如果不做容错，pydantic 解析会报错，整个流程崩掉。
    容错校验器在解析前把类型统一，让系统对模型的"小错误"更宽容。
    """

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

    # mode="before" 表示在 pydantic 做类型检查之前，先跑这个函数处理原始值
    @field_validator("is_clear", mode="before")
    @classmethod
    def _coerce_is_clear(cls, v):
        """把字符串形式的布尔值统一转成真正的 bool。"""
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "是", "清晰")
        return bool(v)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v):
        """确保返回的是字符串列表，处理 None / 单字符串 / 混合类型。"""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v if str(x).strip()]

    @field_validator("clarification_needed", mode="before")
    @classmethod
    def _coerce_clarification(cls, v):
        """确保返回字符串，处理模型有时返回布尔值的情况。"""
        if v is None or isinstance(v, bool):
            return ""
        return str(v)
