# 图的状态定义：主图状态、Agent 子图状态，以及三个状态合并函数(reducer)。
#
# ┌─ 什么是"状态"？ ──────────────────────────────────────────────
# │ LangGraph 的图跑起来时，有一个"状态对象"从一个节点流到下一个节点。
# │ 状态 = 一个装着所有中间数据的字典（运行时其实就是普通 dict）。
# │ 每个节点：收到当前状态 → 干活 → 返回"要更新哪些字段" → 框架合并进状态。
# │ 你可以把它想成一张"随流程不断被填写的表格"。
# │ 下面两个类 State / AgentState 就是这张表格的"模板"（定义有哪些字段）。
# └──────────────────────────────────────────────────────────────

# ── 陌生语法讲解：from typing import ... ──────────────────────────
# typing 是 Python 自带的"类型标注"库。这里拿三样：
#   List[dict]  = "元素是 dict 的列表"，比如 [{...}, {...}]
#   Annotated   = 给一个类型"贴附加说明"的工具（下面 agent_answers 处细讲）
#   Set[str]    = "元素是字符串的集合"，集合天然不重复
from typing import List, Annotated, Set

# ── 陌生对象讲解：MessagesState ──────────────────────────────────
# MessagesState 是 LangGraph【内置】的一个状态基类。它长什么样？
#   它内部已经预先定义好了一个字段：  messages: 消息对象列表
#   并且给这个 messages 字段配好了合并规则(reducer，叫 add_messages)，
#   这个规则支持"往列表里追加新消息"和"用 RemoveMessage 删除某条消息"。
# 我们让下面的 State / AgentState 继承它，就【自动拥有 messages 字段】，
# 不用自己再写。类体里只需补充各自额外需要的字段。
from langgraph.graph import MessagesState

# ── 陌生对象讲解：operator ──────────────────────────────────────
# operator 是 Python 自带模块，把"运算符"变成"函数"。
#   operator.add(3, 5)  等价于  3 + 5  →  8
# 下面把 operator.add 当作某些字段的合并规则，意思是"新旧值相加"（累加）。
import operator


# ============================================================
# 🅰️ 三个 reducer：回答"N 个并行分支同时改一个字段，怎么合并"
#
# 什么是 reducer？
#   图可以有多个分支【并行】运行，都想更新同一个字段。框架需要一个函数来
#   决定"旧值 + 新值 = 合并后的值"，这个函数就叫 reducer。
#   写法：字段类型写成 Annotated[原类型, reducer函数]，框架就会自动调用它。
#   没有 reducer 的字段 = 直接覆盖（谁后写谁说了算）。
# ============================================================

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """agent_answers 字段的 reducer：累积 OR 一键清空。

    入参:
        existing : 目前已有的值（框架自动传入，是旧值）
        new      : 这次节点返回的新值（框架自动传入）
    两种情况：
        ① new 里带 __reset__ 标记 → 返回空列表（清空，用于每轮对话开始时）
        ② 普通 new → existing + new（把新答案追加到末尾）
    """
    # any(可迭代对象) 是 Python 内置函数：里面只要有一个为真，就返回 True。
    # item.get('__reset__') 取字典的 __reset__ 键，没有这个键就返回 None(假)。
    if new and any(item.get('__reset__') for item in new):
        return []                  # 清空：丢掉 existing 的全部内容
    return existing + new          # 累积：列表相加 = 拼接（[1]+[2] 得到 [1,2]）


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """retrieval_keys 字段的 reducer：两个集合求并集，自动去重。

    入参 a、b 都是集合（框架传入旧值和新值）。
    | 是集合的"并集"运算符：{1,2} | {2,3} == {1,2,3}（重复的 2 只留一个）。
    """
    return a | b


def append_unique(existing: List[str], new: List[str]) -> List[str]:
    """retrieved_contexts 字段的 reducer：追加 + 去重 + 保持顺序。

    ── 语法讲透：dict.fromkeys(列表) ──────────────────────────────
    dict 的 key 天然不能重复。dict.fromkeys([...]) 用这个列表建一个字典，
    只保留 key（value 全是 None），重复的 key 自动忽略；再 list(...) 转回
    列表，就得到"去重且保持原顺序"的结果。
        existing + new = ["A","B","B","C"]      ← 有重复
        dict.fromkeys(...) → {"A":None,"B":None,"C":None}
        list(...)          → ["A","B","C"]      ← 去重且保序
    （对比 list(set(...)) 也能去重，但 set 无序、会打乱顺序，所以这里不用它。）
    """
    return list(dict.fromkeys(existing + new))


# ============================================================
# 主图的状态模板（每轮对话全程共享这一份）
# ============================================================
class State(MessagesState):   # 继承 MessagesState → 自动带 messages 字段
    """主图的状态。

    继承来的 messages 字段没写在下面，但它确实存在（对话历史都在里面）。
    这里只补充主图额外要用的字段。每个字段写法是： 名字: 类型 = 默认值
    """
    questionIsClear: bool = False              # 问题清晰吗（rewrite_query 写，路由函数读）
    conversation_summary: str = ""             # 旧对话的滚动摘要（老消息删掉后精华存这）
    originalQuery: str = ""                     # 用户最原始的问题（汇总答案时要用）
    pendingQuery: str = ""                      # 澄清流程里"还没解决的问题"暂存
    pendingClarifications: List[str] = []       # 用户分多次补充的澄清
    rewrittenQuestions: List[str] = []          # 改写后的子问题列表（最多3个，派并行Agent）

    # agent_answers：所有并行 Agent 汇报的答案。
    # Annotated[List[dict], accumulate_or_reset] 的意思是：
    #   这个字段类型是 List[dict]，另外贴一张"合并规则"纸条 = accumulate_or_reset。
    #   框架看到这张纸条，多个分支写入时就调 accumulate_or_reset 来合并（追加，不覆盖）。
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


# ============================================================
# Agent 子图的状态模板（每个并行 Agent 实例各自独立一份）
# ============================================================
class AgentState(MessagesState):   # 同样继承 → 自动带 messages 字段
    """单个 Agent 子图的状态。

    每个并行分支（对应一个子问题）都有自己独立的 AgentState，互不干扰。
    这里的 messages 和主图的 messages 是【两个独立列表】，子图不污染主图历史。
    """
    question: str = ""                          # 这个 Agent 要回答的子问题（Send 传入）
    question_index: int = 0                      # 编号（汇总答案时按它排序，Send 传入）
    context_summary: str = ""                    # 子图内部检索内容的压缩摘要
    final_answer: str = ""                       # 最终答案（只留档，真正流转的是下面的 agent_answers）

    # retrieval_keys：已经"搜过的词 / 取过的父块ID"集合，防止重复检索。
    # Annotated[Set[str], set_union] → 多次写入时用 set_union 求并集去重。
    retrieval_keys: Annotated[Set[str], set_union] = set()

    # retrieved_contexts：检索到的原文文本块（评测用，不发给模型）。
    # Annotated[List[str], append_unique] → 多次写入时用 append_unique 追加去重。
    retrieved_contexts: Annotated[List[str], append_unique] = []

    # 子图自己的 agent_answers：注意【没有 Annotated】，是普通列表、无 reducer。
    # 因为子图里只有一个 collect_answer 节点写它，不存在并行写入。
    # 子图结束时，这里的值会"冒泡"到主图的 agent_answers（那个有 reducer，会追加）。
    agent_answers: List[dict] = []

    # 下面两个计数器：Annotated[int, operator.add] → 合并规则是"加法"。
    # 每圈循环节点返回"本圈新增了多少"，框架用加法自动累加成总数。
    tool_call_count: Annotated[int, operator.add] = 0   # 累计调用了多少次工具
    iteration_count: Annotated[int, operator.add] = 0   # 累计循环了多少圈
