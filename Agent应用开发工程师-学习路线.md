# Agent 应用开发工程师 · 学习路线

> **起点**：能读懂带注释的 LangGraph RAG 项目代码（多智能体并行、reducer、Command、子图、工具调用）。
> **短板**：Python 基础薄弱；没自己从零写过；写完的 Agent **别人用不了**。
> **终点**：一个能上线、能给别人用、能写进简历的**完整 Agent 产品**。
> **周期**：约 2 个月。

---

## 目录

- [〇、先看清全局：你缺的是哪几层](#part-0)
- [一、最该听的一句实话](#part-1)
- [二、阶段 0 · 补三个 Python 短板（1 周）](#part-stage0)
- [三、阶段 1 · FastAPI：让 Agent 变成一个网站（2–3 周）🔥](#part-stage1)
- [四、阶段 2 · PostgreSQL：让它不丢数据（2 周）](#part-stage2)
- [五、阶段 3 · Redis：让它扛得住用（1–2 周）](#part-stage3)
- [六、阶段 4 · Docker：让它能给别人跑（1–2 周）](#part-stage4)
- [七、阶段 5 · 深水区（持续）](#part-stage5)
- [八、明确不用学的东西](#part-skip)
- [附录 A · AGENTS.md 是什么，为什么必须下](#appx-agents)
- [附录 B · 我那个 RAG 项目里已经埋好的伏笔](#appx-scars)
- [附录 C · 产出物总检查表](#appx-checklist)

---

<a id="part-0"></a>
## 〇、先看清全局：你缺的是哪几层

一个**能上线**的 Agent 产品是这样分层的：

```
              ┌──────────────────┐
              │   用户 · 浏览器    │
              └────────┬─────────┘
                       ▼
        ┌──────────────────────────────┐
        │  FastAPI                     │  ❌ 没学
        │  包成 HTTP 接口 · 流式吐字      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  LangGraph Agent              │  ✅ 已经会了
        │  ★ 你现在学的就是这一层 ★       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────┐
  │  向量库    │  │ PostgreSQL │  │  Redis   │
  │  ✅ 会了   │  │   ❌ 没学   │  │  ❌ 没学  │
  │ 存文档·检索│  │ 存对话·用户 │  │缓存·限流 │
  └──────────┘  └────────────┘  └──────────┘

        全部装进 Docker  ❌ 没学
```

**一句话说清每个东西干嘛用的：**

| 技术 | 没有它会怎样 |
|---|---|
| **FastAPI** | 你的 Agent 只能在你自己的终端里跑。**别人用不了，你就没有产品。** |
| **PostgreSQL** | 服务一重启，所有用户的对话历史全没（你现在的 `InMemorySaver` 就是这毛病） |
| **Redis** | 一个用户狂刷接口就能把 API 账单刷爆；Agent 跑 30 秒，HTTP 请求干等到超时 |
| **Docker** | 你本地能跑，换台机器就是一堆环境错误 |

---

<a id="part-1"></a>
## 一、最该听的一句实话

> **能读懂注释精美的代码 ≠ 会写代码。**

这是"精读学习法"的天然副作用。把 `should_compress_context` 读到滚瓜烂熟，和从空文件写出一个能跑的 `/chat` 接口，是**两种完全不同的能力**。

**所以下面每个阶段都有「产出物」。没有产出物的阶段不算学完。别再囤代码笔记了。**

---

<a id="part-stage0"></a>
## 二、阶段 0 · 补三个 Python 短板（1 周）

⚠️ **不要去上「Python 从入门到精通」**，你会在第 30 集放弃。
**只补这三样**，因为它们卡死了 FastAPI 的每一行。而且——**这三样你在 RAG 项目里都见过，只是没搞懂。**

### (1) 类和 `self`

**它解决什么问题**：`collection`（向量库连接）不能让大模型传进来，必须**藏在某个地方**让函数自己取。类就是那个地方。

```python
class ToolFactory:
    def __init__(self, collection):          # 造对象时跑一次
        self.collection = collection         # ⭐ 存在"我自己身上"

    def _search_child_chunks(self, query):
        return self.collection.similarity_search(query)   # ⭐ 随时能取
```

**`self` 不是关键字，就是第一个参数的名字。** Python 自动把「点号左边那个对象」塞进去：

```python
factory._search_child_chunks("上海住宿费")
#   ↑ Python 偷偷翻译成：
#   ToolFactory._search_child_chunks(factory, "上海住宿费")
#                                    ↑↑↑↑↑↑↑ self 就是它
```

🔢 `factory` 对象真实长什么样（一个装东西的盒子）：
```python
factory.__dict__  →  {"collection": <向量库连接>, "parent_store_manager": <父块管理器>}
```

**两个必踩的坑：**
```python
class A:
    def f(query): ...        # ❌ 忘了 self
a = A(); a.f("x")
# TypeError: f() takes 1 positional argument but 2 were given
#            ↑ 多出来的那个就是 self

class B:
    def __init__(self, x):
        x = x                # ❌ 局部变量，函数一结束就没了
        self.x = x           # ✅ 挂在对象上，别的方法才能用
```

### (2) 装饰器 `@`

⭐ **好消息：你已经见过它的「原形」了。** `tools.py` 最后那两行就是装饰器**脱掉糖衣**的样子：

```python
search_tool = tool("search_child_chunks")(self._search_child_chunks)
#              ↑ 一个函数，吃进去一个函数，吐出来一个新对象

# 等价于：
@tool("search_child_chunks")
def _search_child_chunks(query): ...
```

> **装饰器 = 一个「把函数吃进去、吐出一个新函数」的函数。**
> `@X` 只是 `f = X(f)` 的语法糖，没有魔法。

**亲手写一个**（这就是你项目里 `logged_node` 干的事）：

```python
def 打印壳(原函数):
    def 新函数(*args, **kwargs):
        print(f"→ 进入 {原函数.__name__}，参数={args}")
        结果 = 原函数(*args, **kwargs)
        print(f"← 退出 {原函数.__name__}，返回={结果}")
        return 结果
    return 新函数           # ⭐ 返回【函数本身】，不是调用结果

@打印壳
def 加法(a, b):
    return a + b

加法(3, 5)
# → 进入 加法，参数=(3, 5)
# ← 退出 加法，返回=8
```

⭐ **关键**：`@打印壳` 跑完后，`加法` 这个名字**已经不指向原函数了**，它指向 `新函数`，原函数被**包在里面**。

**装饰器有两种，别混：**

| 类型 | 干什么 | 你见过的 |
|---|---|---|
| **① 包一层壳**（改变行为） | 在原函数前后加东西 | `logged_node`（加打印）、`@lru_cache`（加缓存） |
| **② 登记注册**（行为不变） | 只是把函数**记到一张表上** | `@field_validator`（记进校验表）、**`@app.get("/chat")`（记进 FastAPI 路由表）** |

FastAPI 的 `@app.get(...)` 属于第 ② 种——它**不改你的函数**，只是告诉 app「有人访问这个路径时，调这个函数」。

**🔢 `@tool` 的变形链（你项目里真实发生的）：**
```
变形前：普通 Python 函数
   def _search_child_chunks(query: str, limit: int = 5) -> str:
       """在文档中搜索与用户问题相关的片段（子块）…"""
   → 大模型完全不知道它存在
        │ tool(...)(...) 把它吃进去
        │ 读【函数签名】 → 生成 args_schema
        │ 读【docstring】→ 生成 description
        ▼
变形后：StructuredTool 对象
   StructuredTool(
       name="search_child_chunks",
       description="在文档中搜索与用户问题相关的片段（子块）…",  ← docstring 变成了它
       args_schema=<{query: str, limit: int}>,               ← 签名变成了它
       func=<原函数，被包在里面>,
   )
   → 现在能 llm.bind_tools([它]) 交给模型了
```
**这就是为什么 `tools.py` 反复强调「docstring 不是注释，是给模型看的使用手册」。**

### (3) `async` / `await`

**先建立直觉：你的 Agent 90% 的时间在【等】**——等模型返回（2–5 秒）、等向量库、等数据库。这段时间 **CPU 完全空闲**。

**🎨 对照实验（3 个用户同时提问，每个等模型 3 秒）：**

```text
☆ 同步（普通 def）☆
用户A  ████████ 3秒
用户B           ████████ 3秒        ← 干等 A 结束才开始
用户C                    ████████ 3秒
                                     总共 9 秒 ❌

★ 异步（async def + await）★
用户A  ████████ 3秒
用户B  ████████ 3秒                  ← 同时开始！A 在等的时候 CPU 切去伺候 B
用户C  ████████ 3秒
                                     总共 3 秒 ✅
```

⭐ **异步不是"跑得更快"，而是"等的时候不闲着"。**
`await` = 「我要开始等了，CPU 你先去忙别的，好了叫我」。

**⚠️ 最大的坑（AGENTS.md 专门警告的那个）：**

```python
# ❌ 灾难：async 路由里写阻塞调用
@app.post("/chat")
async def chat(req):
    time.sleep(3)                    # ← 没有 await！事件循环被【冻住】
    return ...                       #   这台服务器上【所有】用户全部卡死

# ✅ 正确 1：async 路由 + await 异步版本
@app.post("/chat")
async def chat(req):
    result = await agent_graph.ainvoke(...)   # ⭐ ainvoke = invoke 的异步版

# ✅ 正确 2：干脆写普通 def，FastAPI 自动扔进线程池
@app.post("/chat")
def chat(req):
    result = agent_graph.invoke(...)  # 阻塞，但只堵住线程池一个工人，不影响别人
```

⭐ **一句话记死**：`async def` 里只要出现**一个不带 `await` 的耗时操作**，整个服务就废了。
**拿不准就写普通 `def`**——慢一点，但绝不会全崩。

💡 **LangGraph 里的对应**：`invoke` / `stream`（同步） ↔ `ainvoke` / `astream`（异步）。**带 `a` 前缀就是异步版。** 你项目里全是同步版，接 FastAPI 时要换。

### 阶段 0 · 一周排期

| 天 | 干什么 | 产出物 |
|---|---|---|
| 1–2 | 类和 self | 照着 `ToolFactory` 自己写一个 `NoteFactory` 类：`__init__` 存一个笔记列表，两个方法：加笔记、搜笔记 |
| 3–4 | 装饰器 | 自己写上面那个 `打印壳`，**给 `NoteFactory` 的两个方法加上它**，看打印输出 |
| 5–7 | async/await | 跑一遍上面那个对照实验：用 `asyncio.sleep(3)` 模拟"等模型"，同步版跑一次、异步版跑一次，**亲眼看到 9 秒 vs 3 秒** |

---

<a id="part-stage1"></a>
## 三、阶段 1 · FastAPI：让 Agent 变成一个网站（2–3 周）🔥

**这是整条路线上落差最大的一步。做完你就从"读代码的人"变成"做产品的人"。**

### 主线教程

`https://fastapi.tiangolo.com/tutorial/`

⭐ **FastAPI 是极少数「官方文档 > 所有第三方教程」的框架**，作者亲自写的。**别绕过它去找野教程。**

### 只做这几章，其余全跳过（按顺序）

| 顺序 | 章节 | 为什么 |
|---|---|---|
| 1 | First Steps | 跑起来第一个接口 |
| 2 | Path / Query Parameters | 怎么从 URL 拿参数 |
| 3 | **Request Body** | ⭐ 这就是 Pydantic——**你已经会一半了**（`QueryAnalysis(BaseModel)`） |
| 4 | Response Model | 规定接口吐什么 |
| 5 | **Concurrency and async / await** | ⭐ 官方专门写了一整章，写得极好 |
| 6 | **Dependencies** | ⭐ 把 `llm`、`collection`、`agent_graph` 注进路由 |
| 7 | Custom Response → **StreamingResponse** | ⭐ 流式吐字，LLM 后端的命门 |

### 配套弹药（必备）

下载 **`AGENTS.md`**（见[附录 A](#appx-agents)），学的时候连同 v13.1 提示词一起贴给 AI。

### 用 v13.1 提示词学时，这句话必须写进去

> 「我在做一个 LangGraph RAG Agent 的 HTTP 后端。请所有例子都围绕 `/chat` 接口、`thread_id` 会话隔离、Agent 流式输出来举，**不要用 `/items/{item_id}` 这种玩具例子敷衍我。**」

### 🎁 你已经会的东西（别重新学）

```python
# graph_state.py 里你读过的
agent_answers: Annotated[List[dict], accumulate_or_reset]
#              ↑ 类型            ↑ 纸条 = 合并函数

# FastAPI 的现代写法（AGENTS.md 力推）
PostDep = Annotated[dict, Depends(valid_post_id)]
#         ↑ 类型   ↑ 纸条 = 依赖函数
```
**一模一样的语法。** 都是 `Annotated[类型, 附加物]` = **给类型贴张纸条，框架读纸条办事**：

| 框架 | 纸条上写什么 | 框架读了干嘛 |
|---|---|---|
| LangGraph | reducer | 并行分支写同一字段时，**用它合并** |
| FastAPI | `Depends(...)` | 请求进来时，**用它注入依赖** |

### ✅ 产出物（做不到就没学完）

- [ ] 浏览器打开 `localhost:8000/docs`，看到自动生成的接口文档
- [ ] POST 一个问题，你的 RAG Agent **一个字一个字流式吐回来**
- [ ] 换 `thread_id` = 换一个用户，会话互相独立

---

<a id="part-stage2"></a>
## 四、阶段 2 · PostgreSQL：让它不丢数据（2 周）

**为什么是 PostgreSQL 不是 MySQL**：FastAPI 官方全栈模板用它；LangGraph 的持久化主流用 `langgraph-checkpoint-postgres`；Python 社区整体偏它。（两者 SQL 语法 90% 相同，学哪个都行，但 PostgreSQL 对你更直接有用。）

### 主线教程

`https://sqlmodel.tiangolo.com/`

⭐ **为什么是 SQLModel**：**它是 FastAPI 同一个作者写的，本质就是 Pydantic + 数据库**。你已经会 Pydantic 了，学起来顺水推舟。

**只学**：建表、增删改查、和 FastAPI 用 `Depends` 接起来、Alembic 迁移。

⚠️ AGENTS.md 提醒：用 **SQLAlchemy 2.0 的 async API**（`AsyncSession`）。**在 `async def` 里用同步 ORM = 阻塞事件循环 + 可能死锁连接池。**

### ✅ 产出物

- [ ] 把 `InMemorySaver` 换成 `PostgresSaver`
- [ ] **杀掉服务、重启，对话历史还在**

---

<a id="part-stage3"></a>
## 五、阶段 3 · Redis：让它扛得住用（1–2 周）

不用系统学，**只学两个场景**：

| 场景 | 解决什么 |
|---|---|
| **限流** | 同一个 `thread_id` 一分钟最多问 10 次 → **保护你的 API 账单** |
| **缓存** | 一模一样的问题，直接返回上次的答案 → **省钱 + 秒回** |

**再加一个后台任务队列**（Celery / Arq / RQ），因为：

> AGENTS.md 明确说：`BackgroundTasks` **没有重试、worker 一挂任务就丢**，只适合 <1 秒、丢了也无所谓的活。
> **Agent 一跑 30 秒，必须用 Celery / Arq / RQ。**

### ✅ 产出物

- [ ] 狂刷接口会被限流挡住
- [ ] Agent 跑 30 秒不阻塞 HTTP，前端能查进度

---

<a id="part-stage4"></a>
## 六、阶段 4 · Docker：让它能给别人跑（1–2 周）

只学 `Dockerfile` + `docker-compose.yml`。

### ✅ 产出物

- [ ] 一条 `docker compose up`，Agent + PostgreSQL + Redis + 向量库**全起来**

---

<a id="part-stage5"></a>
## 七、阶段 5 · 深水区（持续）

| 主题 | 说明 |
|---|---|
| **评测（RAGAS）** | 你项目里的 `retrieved_contexts` 就是为它准备的 |
| **可观测性** | LangSmith / Langfuse——看清每次调用花了多少钱、慢在哪 |
| **成本优化** | `MAX_TOOL_CALLS` / `MAX_ITERATIONS` 已经是雏形 |
| **多 Agent / MCP** | 工具生态 |

---

<a id="part-skip"></a>
## 八、明确不用学的东西（别浪费时间）

- ❌ Django / Flask
- ❌ 前端框架（React / Vue）
- ❌ Kubernetes
- ❌ 机器学习理论 / 从零训模型
- ❌ 微服务架构

**不是它们不好，是对"Agent 应用开发工程师"这个具体目标，投入产出比太低。** 真需要了再学。

---

<a id="appx-agents"></a>
## 附录 A · `AGENTS.md` 是什么，为什么必须下

### 它是什么

一个**写给 AI 看的 README**。同样的规则，`README.md` 给人读，`AGENTS.md` 给 AI 读——**版本钉死、Do/Don't 代码块、反模式清单、速查表**，全是机器最容易匹配的格式。

**地址**：`https://github.com/zhanymkanov/fastapi-best-practices/blob/master/AGENTS.md`（点 Raw 保存）

### 为什么你必须下它

**① AI 的训练数据是过期的，它会理直气壮教你写废弃代码。**

**② ⭐ 这文件里有一张表，标题就叫 `Anti-patterns — common AI-agent mistakes`。**
作者原话：**每一条都是我亲眼见过 AI 写出来的真实故障。**

| AI 会写 | 为什么错 | 该写 |
|---|---|---|
| `requests.get(...)` 在 `async def` 里 | 阻塞事件循环，**全服务卡死** | `httpx.AsyncClient` |
| `time.sleep()` 在 `async def` 里 | 同上 | `await asyncio.sleep()` |
| `from jose import jwt` | `python-jose` 无人维护 | `import jwt`（PyJWT） |
| `json_encoders={...}` | Pydantic v2 **已删除** | `@field_serializer` |
| `Field(ge=18, default=None)` | 约束和默认值**自相矛盾** | 二选一 |
| `def f(x = Depends(...))` | 老式写法，有默认值的坑 | `Annotated[T, Depends(...)]` |
| 用 `BackgroundTasks` 跑要紧的活 | **没重试，worker 挂了任务就丢** | Celery / Arq / RQ |
| `async def` 里用同步 ORM | 阻塞循环 + 可能死锁连接池 | `AsyncSession` |
| 路由体里 `catch Exception` | **把 500 变成静默的 200，藏起 bug** | 抓具体异常，抛 `HTTPException` |

**你是「零基础 + 靠 AI 学」，你是这张表精准瞄准的人。**

**③ 它把版本钉死了：**
```
Python 3.11+ ｜ FastAPI 0.115+ ｜ Pydantic 2.7+
SQLAlchemy 2.0（async）｜ PyJWT 2.9+ ｜ ruff 0.6
```

### 怎么用（三步）

1. 下载 `AGENTS.md`
2. 学 FastAPI 的**每一次新对话**，把它连同 v13.1 提示词一起贴进去
3. 加一句：

> 「这是 FastAPI 的现代最佳实践和**反模式清单**。你写的每一段代码，**都要先对照那张「AI 最爱犯的错」表自查一遍**，再给我。」

💡 **这和 v13.1 的铁律 2 是同一个道理**：*「我贴的代码 > 你抓到的网页」*。AGENTS.md 就是你手里那份**唯一权威**。

---

<a id="appx-scars"></a>
## 附录 B · 我那个 RAG 项目里已经埋好的伏笔

**作者是按能上线的标准设计的，但他停在了「能跑但没人能用」。** 每处伏笔都在等一个阶段来兑现：

| 代码里的伏笔 | 在等哪个阶段 |
|---|---|
| `InMemorySaver()` + 注释「进程重启后消失」 | **阶段 2**：换 `PostgresSaver` |
| `config = {"configurable": {"thread_id": "user_123"}}` | **阶段 1**：多用户会话隔离 |
| `retrieved_contexts`「评测用，不发给模型」 | **阶段 5**：接 RAGAS |
| `MAX_TOOL_CALLS` / `MAX_ITERATIONS` | **阶段 3、5**：成本控制 |
| 全套 `invoke`（同步版） | **阶段 1**：换成 `ainvoke` 接异步 Web |

**"到 100"不是再学一个 Agent 框架，是把这五张字条一张张兑现掉。**

### 🔧 顺带：每个状态字段，都是一次翻车的墓碑

作者绝不是先画好图再写的。**每个"看起来多余"的字段，都是一道疤：**

| 字段 | 哪次翻车的墓碑 |
|---|---|
| `iteration_count` / `tool_call_count` | 模型死循环，烧钱 |
| `retrieval_keys` | 模型重复取同一个父块 |
| `context_summary` | token 爆了 |
| `TOKEN_GROWTH_FACTOR` | 压缩阈值死循环 |
| `retrieved_contexts` | 压缩把评测要的原文烧光了 |
| `Command(goto=)` | 一个节点既要改状态又要选路 |
| `agent_answers` + `accumulate_or_reset` | 并行答案互相覆盖 |
| `question_index` | 并行答案顺序乱 |
| `__reset__` | 上一轮答案没清干净 |
| `pendingQuery` / `pendingClarifications` | interrupt 暂停后原问题丢了 |
| 三个 `field_validator` | 国产模型不听话，返回类型不对 |
| `_name_internal_message` | 子图消息污染主图对话历史 |

⭐ **想真正学会，别照着最终代码抄一遍——按疤痕顺序，自己撞一遍：**

| 步骤 | 做什么 | 必须亲眼看到 |
|---|---|---|
| 1 | 写 20 行裸 RAG（**不许用 LangGraph**） | 问一个双话题问题，**看着它答一半** |
| 2 | 加 `bind_tools` + `while` 循环 | **看着它死循环烧钱** |
| 3 | 加预算上限 | 看着它第 8 轮停下来 |
| 4 | 塞 5 个大父块进去 | **看着 token 飙到 4 万** |
| 5 | 加压缩 | **看着 RAGAS 拿不到 contexts**（原文被烧了） |
| 6 | ……然后才引入 LangGraph | 因为你的 `while` 已经乱得管不动了 |

**只有亲眼看到它坏，那个字段才会真正长在你脑子里。**

---

<a id="appx-checklist"></a>
## 附录 C · 产出物总检查表

| 阶段 | 时长 | ✅ 产出物 | 结束时你能说 |
|---|---|---|---|
| **0 · Python 短板** | 1 周 | 自己写的类 + 自己写的装饰器 + 亲眼看到 9 秒 vs 3 秒 | 「装饰器我会自己写了」 |
| **1 · FastAPI** 🔥 | 2–3 周 | `/docs` 能打开；Agent 流式吐字；`thread_id` 隔离会话 | **「我的 Agent 有网址了，你可以打开用」** |
| **2 · PostgreSQL** | 2 周 | `PostgresSaver` 替换成功；重启后历史还在 | 「重启不丢数据了」 |
| **3 · Redis** | 1–2 周 | 限流生效；Agent 扔后台跑不阻塞 | 「不怕被刷爆账单了」 |
| **4 · Docker** | 1–2 周 | `docker compose up` 一键全起 | 「一条命令，别人的机器也能跑」 |
| **5 · 深水区** | 持续 | RAGAS 分数；LangSmith 看板 | 「我知道它好不好、慢在哪、花了多少钱」 |

---

> **总原则**：
> 1. **没有产出物的阶段不算学完。** 别再囤代码笔记了。
> 2. **每次学新东西，先看有没有 `AGENTS.md` 这类"给 AI 的说明书"**，有就先喂给 AI。
> 3. **想真正掌握一个设计，就亲手把它搞崩一次。**
