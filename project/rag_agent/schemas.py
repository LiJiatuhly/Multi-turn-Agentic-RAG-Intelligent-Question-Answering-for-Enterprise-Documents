# 结构化输出的数据模型：定义 rewrite_query 节点期望模型返回的 JSON 结构。
#
# ┌─ 这个文件在整个项目里扮演什么角色？ ──────────────────────────────
# │ 只定义了一个类 QueryAnalysis。它不是"节点"、不干活、不读写状态，
# │ 而是一张【模具/合同】：规定"rewrite_query 这个节点调大模型时，模型必须
# │ 吐出长成什么样的 JSON"。模型吐出的 JSON 会被自动塞进这个模具，变成一个
# │ QueryAnalysis 对象，rewrite_query 再从对象里取三个字段用。
# │
# │ 一句话：schemas.py 负责"约定模型输出的形状"，nodes.py 的 rewrite_query
# │ 负责"拿这个形状里的值去写主图状态"。两者靠下面这三个字段对接。
# └──────────────────────────────────────────────────────────────────
#
# ★ 三个字段 → 谁把它写进主图 State → 主图哪个字段收（数据流转一览）★
#   is_clear             ──rewrite_query 读它──→ 写进 State.questionIsClear
#   questions            ──rewrite_query 读它──→ 写进 State.rewrittenQuestions
#   clarification_needed ──rewrite_query 读它──→ 不清晰时用来生成澄清提示(AIMessage)
#   （写完之后，edges.py 的 route_after_rewrite 读 State.questionIsClear 决定：
#     清晰→遍历 rewrittenQuestions 并行发 Agent；不清晰→去 request_clarification。）
#   想追这条链怎么流转，去 nodes.py 看 rewrite_query，再去 edges.py 看 route_after_rewrite。

# ── 陌生对象讲解：from pydantic import BaseModel, Field, field_validator ──
# pydantic 是一个"数据校验库"，专门干"把一坨原始数据按你定义的规矩检查+转换成对象"。
# 这里拿三样：
#   BaseModel        = "数据模具"的基类。让一个类继承它(class X(BaseModel))，
#                      这个类就自动获得"按字段声明校验/解析/序列化"的能力。
#   Field(...)       = 给某个字段"贴详细说明"的工具，能设默认值、写描述(description)。
#                      这里的 description 不只是注释——with_structured_output 会把它
#                      塞进给模型的 JSON 格式说明里，等于"告诉模型这个字段是干嘛的"。
#   field_validator  = 一个【装饰器】，给某个字段挂一个"预处理/校验函数"。
#                      解析数据时，pydantic 会先调用它，把不规范的原始值"捋顺"再存。
from typing import List
from pydantic import BaseModel, Field, field_validator


class QueryAnalysis(BaseModel):   # 继承 BaseModel → 获得 pydantic 的解析/校验能力
    """🅱️ 问题改写结果的数据结构（rewrite_query 期望模型返回的 JSON 形状）。

    ── 它是怎么被用起来的：with_structured_output 的三步 ─────────────────
    nodes.py 的 rewrite_query 里有这么一句：
        llm_with_structure = llm.with_structured_output(QueryAnalysis, method="json_mode")
    这个 with_structured_output(本类, method="json_mode") 会自动做三件事：
      1. 在发给模型的系统提示里，附上"你必须返回符合这个结构的 JSON"的格式要求
         （字段名、类型、以及上面 Field(description=...) 里写的说明都会带上）；
      2. 用 json_mode 要求模型【只返回 JSON】（智谱不支持强制 tool_choice，故走 json_mode）；
      3. 把模型返回的那串 JSON 文本，自动解析成一个 QueryAnalysis 实例，
         于是 rewrite_query 就能直接写 response.is_clear / response.questions 取值。
    → 提示词那一段的具体措辞在 prompts.py 的 get_rewrite_query_prompt()（末尾"输出格式"）。

    ── 为什么要下面那三个 field_validator（容错校验器）？ ────────────────
    智谱走 json_mode 时并不是 100% 听话，偶尔会：
      · 把 is_clear 返回成字符串 "true"（而不是布尔 true）；
      · 把 clarification_needed 返回成布尔值（而不是字符串）；
      · 把 questions 返回成单个字符串或 null（而不是字符串数组）。
    这些"小错误"若不处理，pydantic 解析时会直接报类型错误 → 整个图这一步崩掉。
    field_validator 在 pydantic 正式做类型检查【之前】先跑，把类型统一捋顺，
    让系统对模型的不规范输出更宽容（宁可自己纠正，也不要整条流程崩）。
    """

    # ── 字段1：is_clear ──────────────────────────────────────────────
    # 问题清不清晰。
    #   Field(default=False, description=...)：
    #     default=False → 模型没给这个字段时，默认按"不清晰"处理（更安全，宁可去追问）。
    #     description   → 这句话会随格式要求发给模型，告诉它该填什么。
    #   【下游】rewrite_query 读 response.is_clear → 写进 State.questionIsClear
    is_clear: bool = Field(
        default=False,
        description="用户的问题是否清晰、可回答。"
    )
    # ── 字段2：questions ─────────────────────────────────────────────
    # 改写后的子问题列表（1~3 个）。
    #   注意这里是 default_factory=list，不是 default=[]。
    #   为什么？因为 [] 是"可变对象"，若用 default=[] 会让所有实例【共享同一个列表】
    #   （Python 经典坑：可变默认值）。default_factory=list 表示"每次新建实例时，
    #   现调用一次 list() 造一个全新的空列表"，各实例互不干扰。
    #   【下游】rewrite_query 读 response.questions → 写进 State.rewrittenQuestions
    #          → route_after_rewrite 遍历它，给每个子问题发一个 Send 启动并行 Agent
    questions: List[str] = Field(
        default_factory=list,
        description="改写后的、自包含的问题列表。"
    )
    # ── 字段3：clarification_needed ──────────────────────────────────
    # 问题不清晰时，说明还要用户补充什么。清晰时是空字符串。
    #   default="" → 是不可变对象(字符串)，可以直接用 default，不需要 factory。
    #   【下游】rewrite_query 读 response.clarification_needed → 不清晰时据此生成
    #          一条 AIMessage(name="clarification") 发给用户（太短则用兜底话术）
    clarification_needed: str = Field(
        default="",
        description="如果问题不清晰，说明还需要用户补充什么。"
    )

    # ── 陌生语法讲解：@field_validator(字段名, mode="before") + @classmethod ──
    # 这两个装饰器叠在函数上，合起来的意思是：
    #   @field_validator("is_clear", mode="before")
    #       → 把下面这个函数注册成"is_clear 字段的校验器"；
    #       → mode="before" 表示"在 pydantic 做类型检查【之前】就跑我"，
    #         所以函数拿到的 v 是模型给的【原始值】（可能是脏的字符串/布尔），
    #         我们把它捋成正确类型 return 出去，pydantic 再拿这个干净值去校验存储。
    #         (对比 mode="after" 是"类型检查之后再跑"，那时 v 已是正确类型了。)
    #   @classmethod
    #       → 校验器是"类方法"，第一个参数是 cls(类本身)而不是 self(实例)，
    #         因为校验发生在"实例还没造出来"的解析阶段，没有 self 可用。
    #   两个装饰器的顺序固定：@field_validator 在上、@classmethod 在下。

    @field_validator("is_clear", mode="before")
    @classmethod
    def _coerce_is_clear(cls, v):
        """把字符串形式的布尔值统一转成真正的 bool。

        例：模型返回 "true"/"是"/"清晰" → 都算 True；其它字符串 → False。
        入参 v = 模型给的原始值；返回 = 一个真正的 bool，交给 pydantic 存进 is_clear。
        """
        # isinstance(v, str)：先判断"模型是不是错给成了字符串"
        if isinstance(v, str):
            # .strip() 去空格、.lower() 转小写后，看它是不是这几个"真值词"之一
            return v.strip().lower() in ("true", "1", "yes", "是", "清晰")
        return bool(v)   # 本来就是布尔或别的类型 → bool() 强转一下兜底

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v):
        """确保返回的是字符串列表，处理 None / 单字符串 / 混合类型。

        三种脏情况都在这里捋顺：
          None          → []             (模型漏给了)
          "单个问题"    → ["单个问题"]    (模型只给了一个字符串，没包成数组)
          [1, "x", ""]  → ["1", "x"]      (混合类型/空串 → 全转字符串并丢掉空的)
        """
        if v is None:                              # 情况①：漏给 → 空列表
            return []
        if isinstance(v, str):                     # 情况②：给成单个字符串
            return [v] if v.strip() else []        #   非空才包成单元素列表，空串→[]
        # 情况③：本来是列表 → 逐个转成字符串，并过滤掉转换后为空白的项
        return [str(x) for x in v if str(x).strip()]

    @field_validator("clarification_needed", mode="before")
    @classmethod
    def _coerce_clarification(cls, v):
        """确保返回字符串，处理模型有时返回布尔值的情况。

        模型偶尔把这个字段错给成布尔 true/false（本该是"要补充什么"的说明文字）。
        None 或 布尔 → 统一当作"没有澄清说明"，返回空字符串 ""；其它 → str() 强转。
        """
        if v is None or isinstance(v, bool):       # 漏给 或 错给成布尔 → 空字符串
            return ""
        return str(v)                              # 其它情况强转成字符串
