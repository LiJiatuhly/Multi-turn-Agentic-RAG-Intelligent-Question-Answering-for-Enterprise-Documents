# 图的状态定义：主图状态、Agent 子图状态，以及三个状态合并函数(reducer)。
#

#   【读】   = 哪个函数把值取出来用
#   🔢      = 这个字段【真实装着什么】—— 有值时长什么样、空的时候长什么样
#            （只知道"它是个字符串集合"没用，得看到里面真的装着 "parent::公司规章制度_p0"）

# （速查：主图节点/子图节点都在 nodes.py，两个路由函数在 edges.py）
#
# 🔢 全文的样例都基于同一个场景，方便串起来看：
#     用户问："上海出差住宿和交通能报多少"
#       → rewrite_query 拆成 2 个子问题
#       → 2 个 Agent 并行去查
#       → aggregate_answers 合成一段回复

# typing 是 Python 自带的"类型标注"库。这里拿三样：
#   List[dict]  = "元素是 dict 的列表"，比如 [{...}, {...}]
#   Annotated   = 给一个类型"贴附加说明"的工具
#                 ⭐ 它是本文件、乃至全项目【最该吃透的一个语法】。
#                    完整讲解在下面 State.agent_answers 那一行（它第一次真正出现的地方）。
#   Set[str]    = "元素是字符串的集合"，集合天然不重复
from typing import List, Annotated, Set

# MessagesState 是 LangGraph【内置】的一个状态基类。它长什么样？
#   它内部已经预先定义好了一个字段：  messages: 消息对象列表
#   并且给这个 messages 字段配好了合并规则(reducer，叫 add_messages)，
#   这个规则支持"往列表里追加新消息"和"用 RemoveMessage 删除某条消息"。
# 我们让下面的 State / AgentState 继承它，就【自动拥有 messages 字段】，
# 不用自己再写。类体里只需补充各自额外需要的字段。
#
# ⭐ 注意：add_messages 也是一张"纸条"（reducer），只不过是框架【替你贴好】的。
#    你自己贴的那三张纸条（下面三个 reducer），和它是完全一回事。
from langgraph.graph import MessagesState

# operator 是 Python 自带模块，把"运算符"变成"函数"。
#   operator.add(3, 5)  等价于  3 + 5  →  8
# 下面把 operator.add 当作两个计数器字段的合并规则，意思是"新旧值相加"（累加）。
# 🔢 它就是这么个东西，没别的：
#     >>> import operator
#     >>> operator.add(3, 5)
#     8
#     >>> operator.add([1,2], [3])     # 对列表来说 + 是拼接
#     [1, 2, 3]
import operator



# 什么是 reducer？
#   图可以有多个分支【并行】运行，都想更新同一个字段。框架需要一个函数来
#   决定"旧值 + 新值 = 合并后的值"，这个函数就叫 reducer。
#   写法：字段类型写成 Annotated[原类型, reducer函数]，框架就会自动调用它。
#   没有 reducer 的字段 = 直接覆盖（谁后写谁说了算）。
#
# ⚠️⚠️ 下面这三个函数，你在【整个项目里搜不到任何一处调用点】。
#     调用它们的是 LangGraph 框架，不是你写的代码。
#     （这就是"你定义了它，但你从来不调用它"—— 最容易懵的一类代码。）
#     它们唯一的"注册方式"，就是被写在 Annotated[...] 的第二个位置上。


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """🅰️ agent_answers 字段的 reducer：累积 OR 一键清空。

    谁会触发它（即谁写 State.agent_answers）：
        - summarize_history（nodes.py）每轮开头发 [{'__reset__': True}] → 走清空分支
        - 各并行子图的 collect_answer（nodes.py）冒泡上来 → 走累积分支（追加）

    入参（⚠️ 这两个参数都是【框架自动传的】，你不用管）:
        existing : 目前已有的值（旧值）
        new      : 这次节点返回的新值

    🔢 情况① 追加（第 1 个并行子图跑完了，第 2 个也跑完，冒泡上来）：
        existing = [{"index": 0,
                     "question": "上海出差住宿费报销上限是多少",
                     "answer": "每天上限 500 元。",
                     "contexts": ["Parent ID: 公司规章制度_p0\\nContent: 上海出差住宿费上限500元/天"]}]
        new      = [{"index": 1,
                     "question": "上海出差交通费怎么报销",
                     "answer": "高铁二等座据实报销。",
                     "contexts": ["Parent ID: 公司规章制度_p1\\nContent: 交通费按高铁二等座据实报销"]}]
        返回      = existing + new
                 = [{"index": 0, ...}, {"index": 1, ...}]      ← 两条都在 ✅

    🔢 情况② 清空（新一轮对话开始，summarize_history 发的"清空卡"）：
        existing = [{"index": 0, ...}, {"index": 1, ...}]      ← 上一轮的旧答案
        new      = [{"__reset__": True}]                       ← ⭐ 这是【暗号卡】，不是真答案
        返回      = []                                          ← 全丢掉，本轮从零开始

    ⭐ 为什么要有"清空"这一路？
       如果不清空，第二轮对话时上一轮的答案还堆在列表里，
       aggregate_answers 会把上轮的旧答案也拿去合成 → 用户得到一坨莫名其妙的回复。
       （这就是函数名里那个 "or_reset" 的全部由来 —— 它是一次翻车的墓碑。）
    """
    # any(可迭代对象) 是 Python 内置函数：里面只要有一个为真，就返回 True。

    #
    # ⚠️ 为什么先判断 `new and`？因为 new 可能是空列表 []，
    #    对空列表跑 any(...) 会返回 False（不出错，但多此一举）。

    #

    #    发出的 {'__reset__': True} 【逐字一致】。改了这里就必须同步改那里，
    #    否则清空卡失效 —— 而且【不报错】，只是上一轮答案永远清不掉（静默出错）。
    if new and any(item.get('__reset__') for item in new):
        return []                  # 清空：丢掉 existing 的全部内容
    return existing + new          # 累积：列表相加 = 拼接（[1]+[2] 得到 [1,2]）


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """🅰️ retrieval_keys 字段的 reducer：两个集合求并集，自动去重。

    谁会触发它（即谁写 AgentState.retrieval_keys）：should_compress_context（nodes.py）
        （它每轮把"这次搜过的词 / 取过的父块ID"并入历史集合）

    入参 a、b 都是集合（框架传入旧值和新值）。
    | 是集合的"并集"运算符：{1,2} | {2,3} == {1,2,3}（重复的 2 只留一个）。

    🔢 真实调用（第 2 轮检索结束时）：
        a（旧的，第1轮记下的） = {"search::上海住宿费报销上限"}
        b（新的，第2轮记下的） = {"search::上海住宿费报销上限",     ← ⚠️ 模型换了个说法又搜了一遍
                                "search::上海出差交通费标准",
                                "parent::公司规章制度_p0"}
        返回 = {"search::上海住宿费报销上限",       ← 重复的自动合并成一个 ✅
               "search::上海出差交通费标准",
               "parent::公司规章制度_p0"}

    ⭐ 为什么必须去重？因为模型【不记得自己搜过什么】，会反复搜同一个词。
       这个集合就是给它记的账本，最后由 compress_context 写进摘要里，
       变成一句"Already executed (do NOT repeat)" 塞回给模型看。
    """
    return a | b


def append_unique(existing: List[str], new: List[str]) -> List[str]:
    """🅰️ retrieved_contexts 字段的 reducer：追加 + 去重 + 保持顺序。

    谁会触发它（即谁写 AgentState.retrieved_contexts）：should_compress_context（nodes.py）
        （它每轮用 _retrieval_contexts(messages) 抽出检索原文块写进来）

    ⭐⭐ 这个 reducer【存在的唯一理由】，是为了抵消 compress_context 的副作用：
        compress_context 会用 RemoveMessage 把 messages 里的检索原文【全部烧光】，
        可 RAGAS 评测又需要"模型看到了哪些原文"。
        → 所以必须在烧光之前，先偷偷把原文备份到这个字段里。
        （这是"某个东西存在，是为了抵消另一处代码的副作用"的活标本。）

    ── 语法讲透：dict.fromkeys(列表) ──────────────────────────────
    dict 的 key 天然不能重复。dict.fromkeys([...]) 用这个列表建一个字典，
    只保留 key（value 全是 None），重复的 key 自动忽略；再 list(...) 转回
    列表，就得到"去重且保持原顺序"的结果。
        existing + new = ["A","B","B","C"]      ← 有重复
        dict.fromkeys(...) → {"A":None,"B":None,"C":None}
        list(...)          → ["A","B","C"]      ← 去重且保序
    （对比 list(set(...)) 也能去重，但 set 无序、会打乱顺序，所以这里不用它。）

    🔢 真实调用：
        existing = ["Parent ID: 公司规章制度_p0\\nFile Name: 规章制度.pdf\\nContent: 上海出差住宿费上限500元/天"]
        new      = ["Parent ID: 公司规章制度_p0\\nFile Name: 规章制度.pdf\\nContent: 上海出差住宿费上限500元/天",  ← ⚠️ 重复！模型又搜了一遍
                    "Parent ID: 公司规章制度_p1\\nFile Name: 规章制度.pdf\\nContent: 交通费按高铁二等座据实报销"]
        返回      = [住宿那条, 交通那条]         ← 重复的被丢掉，顺序保持不变 ✅

    📎 隐形合同：这里存的字符串，格式是 tools.py 里两个工具拼出来的
       "Parent ID: xxx\\nFile Name: yyy\\nContent: zzz"，
       由 nodes._retrieval_contexts 按同样格式反向解析出来。
       改了 tools.py 的格式，这里存的东西就会错乱。
    """
    return list(dict.fromkeys(existing + new))



# 主图的状态模板（每轮对话全程共享这一份）

class State(MessagesState):   # 继承 MessagesState → 自动带 messages 字段
    """主图的状态。

    继承来的 messages 字段没写在下面，但它确实存在（对话历史都在里面）。
    这里只补充主图额外要用的字段。每个字段写法是： 名字: 类型 = 默认值

    ⚠️ 这里的 "= 默认值" 是个【假默认值】！
       TypedDict 类的默认值【根本不会进实例】—— 运行时流动的是一个普通 dict，
       字段没被写过就是【压根不存在这个键】。这就是为什么全项目到处都是
       state.get("xxx", 兜底值) 而不是 state["xxx"] —— 直接取会 KeyError。
       （例外：messages 一定存在，所以 nodes.py 里敢直接写 state["messages"]。）
    """

    #   作用：用户能看到的真实对话历史（跨轮对话靠 checkpointer 一直保存）。
    #   【写】① chat 入口：追加用户的 HumanMessage
    #        ② summarize_history：发 RemoveMessage 删掉过老的消息
    #        ③ rewrite_query：给澄清相关消息打 name 标签（防止被当普通历史）
    #        ④ aggregate_answers：追加最终回复 AIMessage + 删旧消息

    #
    #   🔢 有值（一轮问答结束后）：
    #     [HumanMessage(content="上海出差住宿和交通能报多少"),

    #      ↑ 注意这两条【都没有 name 属性】—— 这是"真实对话"的标志
    #
    #   🔢 中间态（并行子图跑完、aggregate_answers 还没清理时，是一锅粥）：


    #      ToolMessage(content="Parent ID: 公司规章制度_p0\\n…",


    #
    #   🔢 空态：[]（全新会话，用户还没说话）
    #
    #   ⚠️⚠️ 注意：并行子图的 messages 默认【也会流回】这里！
    #      靠 _name_internal_message 给子图消息打 name 标签、
    #      _is_plain_conversation_message 把带 name 的过滤掉，
    #      才保证子图的"我要调工具/工具结果"不污染真实对话历史。
    #      ⭐ 这就是"子图不污染主图"的全部原理 —— 全项目最隐蔽的一条数据流。

    questionIsClear: bool = False
    #   问题清晰吗。
    #   【写】rewrite_query（清晰=True / 不清晰=False）

    #
    #   🔢 True  ← 用户问"上海出差住宿和交通能报多少"（明确，能直接检索）
    #   🔢 False ← 用户问"报销怎么弄"（太含糊：什么报销？差旅？办公用品？）
    #   ⚠️ 无 reducer → 直接覆盖。每轮 rewrite_query 重写一次。

    conversation_summary: str = ""
    #   旧对话的滚动摘要（老消息被删掉后，精华浓缩存这里）。
    #   【写】summarize_history

    #
    #   🔢 有值（聊了好几轮之后）：
    #     "用户在咨询差旅报销规则。已问过：上海住宿费上限500元/天，需正规发票。"
    #   🔢 空态："" （第一轮对话，还没有历史可摘要）
    #   ⚠️ 无 reducer → 直接覆盖（新摘要已经把旧摘要合并进去了，所以覆盖是对的）

    originalQuery: str = ""
    #   用户最原始的问题（若走过澄清，则是"原问题 + 历次澄清"合并后的完整问题）。
    #   【写】rewrite_query（清晰时写入；不清晰时清空为 ""）
    #   【读】aggregate_answers（汇总时作为"用户最初问的"发给模型）
    #
    #   🔢 没走澄清："上海出差住宿和交通能报多少"
    #   🔢 走过澄清："报销怎么弄 差旅 住宿费"   ← 原问题 + 用户两次补充，拼起来的
    #   🔢 空态：""
    #

    #      因为要让模型知道【用户最初真正想问什么】，才能把几段子答案组织成一段自然的话。
    #      发拆分后的子问题，模型会把回复也写得支离破碎（"关于问题1…关于问题2…"）。

    pendingQuery: str = ""
    #   澄清流程的"记忆"：问题不清晰、要暂停等用户补充时，先把原问题存这，免得暂停后丢了。

    #   【读】rewrite_query（下一轮用户补充后再进来时读它，和新澄清拼成完整问题）

    #
    #   🔢 有值："报销怎么弄"       ← 图暂停了，这句原问题被存着，等用户补充
    #   🔢 空态：""                ← 问题已经问清楚了，澄清流程结束，清空
    #
    #   ⚠️ 它存在的唯一理由：interrupt 一暂停，invoke() 就返回了。
    #      用户下次再发消息时，是一次【全新的调用】—— 原问题早就不在 messages 的末尾了。
    #      没有这个字段，澄清机制就是废的。（这是一次翻车的墓碑。）

    pendingClarifications: List[str] = []
    #   用户分多次补充的澄清，逐条累积成列表。
    #   【写】rewrite_query（不清晰时 = 旧澄清列表 + 这次输入；问清楚后清空为 []）

    #
    #   🔢 有值：["差旅", "住宿费"]     ← 用户被追问了两次，各补充了一句
    #   🔢 空态：[]                    ← 问清楚了，清空
    #

    #      每次是 rewrite_query 自己算好"旧列表 + 新输入"再整个覆盖进去。

    rewrittenQuestions: List[str] = []
    #   改写后的子问题列表（最多 3 个，用来派并行 Agent）。
    #   【写】rewrite_query（清晰时=改写/拆分后的 1~3 个子问题；不清晰时=[]）

    #
    #   🔢 有值（用户问"上海出差住宿和交通能报多少"，被拆成 2 个）：
    #     ["上海出差住宿费报销上限是多少",
    #      "上海出差交通费怎么报销"]
    #     ⭐ 列表里有几个元素，就会并行启动几个 Agent。这是 fan-out 的源头。
    #   🔢 空态：[]（问题不清晰，还没拆）

    # ⭐⭐ agent_answers —— Annotated 第一次真正出现，全项目最该吃透的一个语法
    # 【一句话】给一个类型「贴一张纸条」。
    #          Python 本身【完全不看】这张纸条，但【框架会看】，照着纸条办事。
    #


    #     ①字段名      ②"贴纸条"    ③真正的类型      ④纸条内容      ⑤默认值
    #                    的工具                    (函数名，          (假的！
    #                                              ⚠️没有括号)        见类 docstring)
    #   ⚠️⚠️ 纸条上写的是 accumulate_or_reset，【没有括号】—— 这是最关键的一点：
    #      accumulate_or_reset()  有括号 = "现在就执行它，把返回值贴上去"
    #      accumulate_or_reset    没括号 = "把【函数本身】贴上去，等框架以后来调"
    #   ⭐ 所以你【永远搜不到】任何一处 accumulate_or_reset(...) 的调用点。
    #      调用它的人是 LangGraph 框架，不是你。
    #      —— "你定义了它，但你从来不调用它"，这是全项目最容易懵的一类代码。
    #

    #   也就是说：把纸条撕掉，程序照跑，Python 一点都不在乎。
    #   🔬 真跑一下：
    #       >>> from typing import Annotated
    #       >>> Annotated[int, "随便写点啥"]

    #       >>> def f(x: Annotated[int, "废话"]): return x + 1
    #       >>> f(3)

    #   ⭐ 结论：纸条【纯粹是给框架看的】。Python 只负责当个信使，原样保存。
    #
    #   场景：2 个并行 Agent 同时往 agent_answers 里写自己的答案
    #

    #     Agent0 写 [{"index":0, "answer":"每天上限 500 元。"}]
    #     Agent1 写 [{"index":1, "answer":"高铁二等座据实报销。"}]
    #     框架看到纸条 → 调 accumulate_or_reset(旧值, 新值) → 【追加】

    #
    #   ☆ 没贴纸条 ☆  agent_answers: List[dict]



    #     用户问"住宿和交通能报多少"，只拿到交通的答案。
    #     ⚠️ 而且【不报错】—— 静默出错，最难查的那一种。
    #
    #   ⭐ 这一张纸条，就是"两个答案都在"和"丢一个"的全部差别。
    #
    #   ① 你写代码时（就是现在这一行）
    #        Annotated[List[dict], accumulate_or_reset]
    #        纸条订上去了                             → ❗还没有任何人调用它
    #                    │
    #                    ▼
    #   ② graph_builder.compile() 时（graph.py 里）
    #        框架把纸条【撕下来】，记进一本"合并规则手册"  → ❗仍然没有调用它
    #        （只做这一次，之后再也不看类定义了）
    #                    │
    #                    ▼
    #   ③ 某个节点 return {"agent_answers": [...]} 时
    #        框架翻手册 → 查到 accumulate_or_reset
    #        → 执行 accumulate_or_reset(旧值, 新值)      → ★唯一的调用点★
    #
    #   ⭐ 第 ② 步那句"仍然没有调用它"，比"发生了什么"更能解开困惑。
    #      很多人以为 compile 时会跑一遍 reducer —— 不会。它只是被【登记】了。
    #






    #                         ↑类型        ↑纸条=合并函数  → 框架用它【合并状态】
    #   FastAPI   : Annotated[dict,       Depends(get_db)]
    #                         ↑类型        ↑纸条=依赖函数  → 框架用它【注入依赖】
    #   ⭐ 一模一样的 Annotated[类型, 附加物]。都是"给类型贴纸条，让框架读纸条办事"。
    #      你学 FastAPI 时会以为 Depends 是全新的东西 —— 不是，你已经会了。

    #        ② 各并行子图的 collect_answer（nodes.py）：结束时冒泡追加自己的答案
    #   【读】aggregate_answers（nodes.py）：按 index 排序后合成最终回复
    #   🔢 有值（2 个并行子图都跑完了）：
    #     [

    #          "question": "上海出差交通费怎么报销",
    #          "answer": "高铁二等座据实报销。",

    #         {"index": 0,
    #          "question": "上海出差住宿费报销上限是多少",
    #          "answer": "每天上限 500 元。",

    #     ]
    #     ⚠️⚠️ 顺序是【乱的】—— 哪个子图先跑完，哪个先被追加进来，和问题顺序无关。

    #          ⭐ index 这个字段存在的【全部理由】就是这一行排序。
    #
    #   🔢 空态：[]（每轮对话开头，被 summarize_history 的 __reset__ 清空）
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []



# Agent 子图的状态模板（每个并行 Agent 实例各自独立一份）

class AgentState(MessagesState):   # 同样继承 → 自动带 messages 字段
    """单个 Agent 子图的状态。

    每个并行分支（对应一个子问题）都有自己独立的 AgentState，互不干扰。
    这里的 messages 和主图的 messages 是【两个独立列表】，子图不污染主图历史。

    ── 👯 和主图 State 的近亲对比（长得像，但地盘完全不同）────────────
    | 维度         | State（主图）                  | AgentState（子图）              |
    |--------------|--------------------------------|--------------------------------|
    | 一句话       | 一整轮对话的总账本              | 一个子问题的私人草稿纸           |
    | 有几份       | 每轮对话【一份】                | 每个并行 Agent 【各一份】        |
    | messages 里是| 用户能看到的真实对话             | "我要调工具"+工具返回的原文       |
    | 独有字段     | pendingQuery、rewrittenQuestions| retrieval_keys、context_summary |
    | agent_answers| 有 reducer（并行写，要追加）     | 无 reducer（单点写，直接赋值）    |
    | 为什么不能合并| 主图不该看见子图的工具调用细节；  |                                |
    |              | 子图不该看见别的子问题的答案      |                                |
    """

    #   作用：这个 Agent 的私人草稿纸。被 Send 创建时传入的是空列表 []。




    #
    #   🔢 逐圈演变（真实轨迹，这个子图负责"住宿费"这个子问题）：

    #     orchestrator 第1圈后：
    #       [HumanMessage("上海出差住宿费报销上限是多少"),            ← 子问题

    #                  tool_calls=[{"name": "search_child_chunks",
    #                               "args": {"query": "上海住宿费报销上限"},
    #                               "id":   "call_abc123"}])]        ← ⭐ 这次调用的唯一编号
    #     tools 执行完：


    #                     name="search_child_chunks")
    #     orchestrator 第2圈（模型看完材料，觉得够了）：

    #

    #      ⚠️ 原文全被 RemoveMessage 烧光了，精华挪进了 context_summary。

    #
    #   ⭐ AIMessage 和 ToolMessage 是【一问一答】：
    #      模型自己碰不到向量库，它只能"说话"。整条链是：
    #        模型说"我要搜"（AIMessage.tool_calls）
    #          → 框架 ToolNode 真的去执行
    #            → 结果包成 ToolMessage 塞回 messages
    #      两者靠 id / tool_call_id 配对。

    question: str = ""
    #   这个 Agent 要回答的子问题。
    #   【写】route_after_rewrite（edges.py）里的 Send 传入

    #
    #   🔢 "上海出差住宿费报销上限是多少"     ← 0 号子图拿到的
    #   🔢 "上海出差交通费怎么报销"           ← 1 号子图拿到的（另一个独立实例）

    #      不是别的节点写的，是【创建这个子图时就注入的初始值】。

    question_index: int = 0
    #   编号（汇总答案时按它排序）。
    #   【写】route_after_rewrite（edges.py）里的 Send 传入

    #
    #   🔢 0  ← 第一个子问题（enumerate 的编号）
    #   🔢 1  ← 第二个子问题
    #
    #   ⭐ 它存在的唯一理由：并行子图【谁先跑完谁先冒泡】，agent_answers 里的顺序是乱的。
    #      没有 index，用户看到的答案顺序就是随机的。

    context_summary: str = ""
    #   子图内部检索内容的压缩摘要（检索循环太长时把原文压成这段，腾出 token）。
    #   【写】compress_context

    #        compress_context（读旧摘要合并成新摘要）、fallback_response
    #
    #   🔢 有值（压缩过一轮之后）：
    #     "已知：上海出差住宿费上限 500 元/天，需提供正规发票。
    #
    #      ---
    #      **Already executed (do NOT repeat):**
    #      Parent chunks retrieved:
    #      - 公司规章制度_p0
    #      Search queries already run:
    #      - 上海住宿费报销上限"
    #      ↑⭐ 注意末尾那段"已执行清单"—— 它是从 retrieval_keys 生成的，
    #         专门喂回给模型看，让它别再重复搜同一个词。
    #
    #   🔢 空态：""（compress_context 还没跑过 → 这个字段压根不存在这个键）

    #         .get(默认"") 防 KeyError，.strip() 防"纯空白字符串被当成有内容"。
    #         —— 这就是那些"看不懂的兜底写法"存在的唯一理由：应付空态。

    final_answer: str = ""
    #   最终答案。
    #   【写】collect_answer
    #   【读】❗没有任何函数读它，纯留档（真正流转到主图的是下面的 agent_answers）
    #
    #   🔢 "每天上限 500 元。"          ← 正常答完
    #   🔢 "无法生成答案。"             ← 兜底话术（模型只调工具没说话时的异常态）
    #   ⚠️ 删掉这个字段，程序照跑，什么都不会坏。它纯粹是给人看的留档。

    # retrieval_keys：已经"搜过的词 / 取过的父块ID"集合，防止重复检索。



    #
    #   🔢 有值（跑了两轮之后）：
    #     {

    #         "parent::员工手册_p3",

    #         "search::出差补贴标准",
    #     }
    #   🔢 空态：set()（一次工具都还没调）
    #
    #   ⭐ 前缀的作用：防止"父块ID"和"搜索词"万一撞名，被当成同一个东西混进集合。
    #      一个 5 秒钟的决定，但没有它就会【静默出错】。
    #
    #   🔢 变形前后（should_compress_context 里的加工过程）：
    #     模型给的原始参数   tc["args"] = {"query": "上海住宿费报销上限"}
    #              │  加 "search::" 前缀，标明"这是一次搜索"
    #              ▼
    #     存进集合的暗号     "search::上海住宿费报销上限"
    retrieval_keys: Annotated[Set[str], set_union] = set()

    # retrieved_contexts：检索到的原文文本块（评测用，不发给模型）。



    #
    #   🔢 有值：



    #
    #   ⭐⭐ 这个字段【存在的唯一理由】，是为了抵消 compress_context 的副作用：

    #      可 RAGAS 评测又需要"模型看到了哪些原文"。所以必须提前备份到这里。
    #


    #      改了 tools.py 的格式或"暗号"词，这里就会存进一堆垃圾。
    retrieved_contexts: Annotated[List[str], append_unique] = []


    # 因为子图里只有一个 collect_answer 节点写它，不存在并行写入，直接赋值就行。

    #   【写】collect_answer（本子图唯一写入点，所以不需要 reducer）

    #
    #   🔢 有值（这个子图跑完了）：
    #     [{"index": 0,
    #       "question": "上海出差住宿费报销上限是多少",
    #       "answer": "每天上限 500 元。",

    #     ⚠️ 永远【只有一条】—— 一个子图只回答一个子问题。
    #   🔢 空态：[]（collect_answer 还没跑）
    #
    #   ⭐⭐ 全项目最精妙的一处设计（"两个同名字段，reducer 不同"）：


    #      同一个名字，冒泡跨过子图边界的那一刻，合并规则就换了。
    agent_answers: List[dict] = []

    # 下面两个计数器：Annotated[int, operator.add] → 纸条上写"加法"。
    # 每圈循环节点返回"本圈新增了多少"，框架用加法自动累加成总数。
    #
    # ⭐ 关键理解：节点返回的【不是新总数，是增量】！


    #    框架看到纸条上的 operator.add，自动执行 operator.add(旧总数, 1)。
    #
    #   🔢 tool_call_count 逐圈演变（这个子图跑了 3 圈）：



    #   🔢 空态：0
    #   tool_call_count：【写】orchestrator（每轮 += 本轮工具调用数）

    tool_call_count: Annotated[int, operator.add] = 0   # 累计调用了多少次工具

    #   🔢 iteration_count 逐圈演变：0 → 1 → 2 → 3（每圈固定 +1）
    #   🔢 空态：0
    #   iteration_count：【写】orchestrator（每轮 +1）

    #
    #   🎨 对照实验（把 operator.add 这张纸条撕掉会怎样）：


    #                                                     → 永远到不了 8 → 💸 无限循环烧钱 ❌
    #     ⭐ "删了会怎样"从来不是修辞问句 —— 这是一个可以真跑出来的实验。
    iteration_count: Annotated[int, operator.add] = 0   # 累计循环了多少圈