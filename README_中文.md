# 智能体 RAG 中文问答助手（接入智谱 GLM）

一个「会自己查资料」的中文 PDF 问答助手：把 PDF 丢进去，它自动切块、建立向量库；你用中文提问，它会自己决定去库里检索相关内容，再用中文回答并附上来源。

底层用 **LangGraph** 编排一个智能体，聊天与词嵌入都走 **智谱 GLM**。

> ✅ 已在 Python 3.12 下**真实端到端跑通**：能索引中文文档、正确检索、用中文作答并标注来源。

---

# ⚠️ 第 0 件事：填入你自己的 API Key

`project/.env` 里需要填你的智谱密钥：
1. 打开 https://open.bigmodel.cn/ 登录 → 右上角「API 密钥」（ https://open.bigmodel.cn/usercenter/apikeys ）。
2. 生成一个 API Key 并复制。
3. 填进 `.env`（见下面第③步）。

---

# 它是怎么工作的（读一遍，帮助排错）

```
你的PDF ──切成小块和大块──▶ 用智谱embedding变成向量 ──▶ 存进本地向量库(Qdrant)
                                                                    │
你提问 ──▶ 模型改写问题 ──▶ 模型【自己决定】调用"搜索工具"去库里找 ──▶ 找到相关块
                                                                    │
                          ◀── 模型读了这些块，用中文写出答案(带来源) ──┘
```

- 「模型自己决定调用搜索工具」用的是**函数调用(tool calling)**——这是它「会自己查」的关键。
- 数据都存在项目根目录自动生成的 `qdrant_db / parent_store / markdown_docs` 三个文件夹里，删掉它们就等于清空知识库。

---

# 项目里每个文件在做什么

**顶层**

| 文件 | 作用 |
|---|---|
| `project/app.py` | 启动入口：加载 .env、创建并启动 Gradio 界面 |
| `project/config.py` | 全局配置：模型、接口地址、检索/切块/预算等所有参数都在这里改 |
| `requirements.txt` | 依赖清单 |
| `check_syntax.py` | 语法自检脚本：启动前逐个检查文件有没有语法错误（可选，用法见《运行与自检指南.md》） |
| `project/.env` | 填 API Key、接口地址、模型名的地方 |

**`project/rag_agent/`（智能体核心）**

| 文件 | 作用 |
|---|---|
| `graph.py` | 组装两张图：内层「Agent 子图」(检索循环) + 外层「主图」(改写→并行→汇总) |
| `graph_state.py` | 定义图在运行时携带的状态，以及几个状态合并函数(reducer) |
| `nodes.py` | 图上每个方框的具体逻辑：历史摘要、问题改写、检索编排、上下文压缩、兜底回答、答案汇总 |
| `edges.py` | 条件路由：问题清不清晰走哪条路、Agent 该继续搜还是收尾 |
| `tools.py` | 两个检索工具：`search_child_chunks`(搜小块)、`retrieve_parent_chunks`(取大块)。工具说明会发给模型看 |
| `prompts.py` | 6 个中文系统提示词（摘要/改写/编排/兜底/压缩/汇总） |
| `schemas.py` | 「问题改写」这一步的结构化输出数据模型（含类型容错） |

**`project/db/`（数据层）**

| 文件 | 作用 |
|---|---|
| `vector_db_manager.py` | 向量库管理：用智谱 embedding 做稠密向量 + 本地 BM25 稀疏向量 + Qdrant 混合检索 |
| `parent_store_manager.py` | 父块存储：把大块以 JSON 文件存本地，靠文件名当主键读写 |

**`project/core/`（系统装配）**

| 文件 | 作用 |
|---|---|
| `rag_system.py` | 总装：创建向量库、智谱聊天模型、工具，编译成 Agent 图 |
| `document_manager.py` | 文档处理：接收上传的 PDF/MD，转换、切块、写入向量库和父块存储 |
| `chat_interface.py` | 聊天流处理：把图的流式输出转成 Gradio 能显示的消息 |
| `observability.py` | 可观测性(可选)：接入 Langfuse 追踪，默认关闭 |
| `execution_logger.py` | 调试日志：终端彩色打印每步输入输出，默认关闭 |

**`project/ui/`（界面）**

| 文件 | 作用 |
|---|---|
| `gradio_app.py` | Gradio 界面：文档上传标签页 + 对话标签页 |
| `css.py` | 界面样式 |

**`project/` 其他**

| 文件 | 作用 |
|---|---|
| `document_chunker.py` | 文档切块器：把 Markdown 切成父块(大块,给上下文)和子块(小块,给检索)，子块靠 parent_id 关联回父块 |
| `utils.py` | 通用工具：PDF 转 Markdown、清空目录、估算 token 数 |

---

# 数据流详解（每步进什么、出什么）

> 🟦 程序拼死的　🟨 模型生成的　🟩 工具返回的。例子文档：公司规章制度.pdf；例子问题：上海出差酒店住宿每晚最多能报销多少？

## A. 建索引：一份 PDF 怎么进知识库

```text
公司规章制度.pdf
   │  ① add_documents()  总调度
   ▼
② pdf_to_markdown()   PDF → Markdown 文本
   出： "# 员工报销制度\n## 差旅费报销标准\n一线城市…每晚上限为六百元…"
   │
   ▼
③ create_chunks_single()   按标题切 → 合并过小 → 拆分过大
   出： 2 个「父块」，每块约 2000~4000 字
   │
   ▼
④ __create_child_chunks()   给父块编号 + 切子块
   父块： parent_id="公司规章制度_p0"，正文=整段报销标准
   子块： content="一线城市…六百元…"(约500字)，metadata={parent_id:"公司规章制度_p0"}
   │                                        └── 子块带着 parent_id，这是分层检索的命门
   ├────────────► ⑤ save_many()      父块存成 parent_store/公司规章制度_p0.json
   │
   ▼
⑥ embed_documents()   子块文本 → 智谱向量
   进： ["一线城市…六百元…", …]
   出： [[0.011, -0.023, …共2048个数], …]
   │
   ▼
⑦ collection.add_documents()   子块+向量写入 qdrant_db/   →  知识库就绪
```

## B. 提问：一个问题怎么变成「六百元」

```text
你输入：上海出差酒店住宿每晚最多能报销多少？
   │
   ▼
【2】rewrite_query()  判断清晰度 + 改写（调一次模型）
   ┌─ 先把「上下文」拼出来（代码里叫 context_section）───────────────┐
   │ conversation_summary = ""   ← 第一轮无历史，跳过                 │
   │ recent_messages      = []   ← 无近期对话，跳过                   │
   │ current_query = "上海出差酒店住宿每晚最多能报销多少？"           │
   │        ↓ 把非空部分用 \n\n 拼起来                                │
   │ context_section = "User Query:\n上海出差酒店住宿每晚最多能报销多少？" │
   └──────────────────────────────────────────────────────────────┘
   🟦 发给模型：System=[改写提示词] ＋ Human=上面的 context_section
   🟨 模型返回：{"is_clear":true, "questions":["上海地区酒店住宿费报销上限"], "clarification_needed":""}
   （若这是第二轮追问"那北京呢？"，context_section 里会多出 Recent Conversation 段，
     模型就能把"那北京呢"补全成"北京地区酒店住宿费报销上限"）
   │
   ▼
【R】route_after_rewrite()  清晰 → 返回 [Send("agent", {question:"上海地区酒店住宿费报销上限"})]
   │   （1 个问题 = 1 个并行 Agent；进入下面的检索循环）
   ▼
┌──────────────────── Agent 子图（检索循环）────────────────────┐
│ 【3】orchestrator() 第1圈                                       │
│   🟦 发给模型：System=[编排提示词] ＋ Human="上海地区酒店住宿费报销上限"│
│                ＋ Human="你必须先调用 'search_child_chunks'…"    │
│   🟨 模型返回：tool_calls=[{name:"search_child_chunks",          │
│                            args:{query:"上海 住宿费 报销上限"}}] │
│         │                                                       │
│         ▼ 【R】route → "tools"（有工具调用+预算够）              │
│ 【4】search_child_chunks()  内部把 query 变向量 → 混合检索        │
│   🟩 工具返回（作为 ToolMessage 塞回消息）：                     │
│      Parent ID: 公司规章制度_p0                                  │
│      File Name: 公司规章制度.pdf                                 │
│      Content: 一线城市（北京、上海…）每晚上限为六百元…           │
│         │                                                       │
│         ▼ 【R】should_compress_context()  记下搜过的词；          │
│              token 没超 → goto="orchestrator"（回去再看一眼）    │
│ 【5】orchestrator() 第2圈  这次消息里带着上面搜到的内容           │
│   🟨 模型返回（这次是文字，没有工具调用）：                      │
│      "上海出差酒店住宿每晚最多报销六百元。参考来源：\n- 公司规章制度.pdf" │
│         │                                                       │
│         ▼ 【R】route → "collect_answer"（没有工具调用了）        │
│ 【6】collect_answer()  打包：agent_answers=[{index:0, answer:"…六百元…"}] │
└────────────────────────────────────────────────────────────────┘
   │  （子图结束，答案冒泡回主图；多个并行 Agent 各追加一条）
   ▼
【7】aggregate_answers()  把所有答案合成一个（调一次模型）
   🟨 最终答案：上海出差酒店住宿每晚最多报销六百元。参考来源：\n- 公司规章制度.pdf
   │
   ▼
界面流式显示最终答案
```

> 这两张图最该看的是 B 图里那个虚线框：**「上下文」不是抽象概念，就是一个拼接的字符串**——摘要 + 近期对话 + 当前问题，非空的部分用 `\n\n` 连起来，每段带个标签。第一轮前两块是空的，所以只剩问题本身。

---

# 准备工作：安装 Python 3.12（最关键的一步，务必做对）

> ⚠️ **本项目依赖要求 Python ≥ 3.10。用 3.9 或更老，装依赖会报
> `No matching distribution found for langchain-openai`——这不是代码问题，是 Python 太老。**
> 装 3.12 是**下载安装程序、双击安装**，不是敲命令。这一步没做，后面都白搭。

**第 0.1 步 · 先看看你电脑上有没有 3.12**
在 PowerShell 运行：
```powershell
py -3.12 --version
```
- 显示 `Python 3.12.x` → 已经装过了，**直接跳到下面第 0.3 步**。
- 报错（`not found` 或 `py 不是命令`）→ 没装，做第 0.2 步。

**第 0.2 步 · 安装 Python 3.12（不用卸载旧版本，可以共存）**

方式 A（最省事，一条命令）——如果有 winget：
```powershell
winget install -e --id Python.Python.3.12
```
装完**关掉 PowerShell，重新开一个**让它生效。

方式 B（手动下载）：
1. 打开 https://www.python.org/downloads/release/python-3129/
2. 拉到最底下，点 **"Windows installer (64-bit)"** 下载。
3. 双击安装，**第一屏最下面的 "Add python.exe to PATH" 一定要打勾 ✅**，再点 "Install Now"。
4. 装完**关掉 PowerShell 重新开一个**，运行 `py -3.12 --version` 确认显示 `Python 3.12.x`。

**第 0.3 步 · 确认无误**
`py -3.12 --version` 能显示 `Python 3.12.x` 就绪。（`py --list` 可看你装了哪些版本。）

> 📌 注意：你系统里可能同时有 3.9 和 3.12，直接敲 `python` 可能还是旧的 3.9。
> 所以下面第④步**一定要用 `py -3.12` 来建虚拟环境**，别用 `python`。

---

# 第①步：解压项目
把压缩包解压，得到 `agentic-rag-cn` 文件夹（里面有 `project` 和 `requirements.txt`）。建议解压到桌面等简单路径。

# 第②步：在项目文件夹里打开 PowerShell
1. 进入 `agentic-rag-cn` 文件夹（能看到 `requirements.txt` 的那一层）。
2. **按住 Shift + 右键** 空白处 →「在终端中打开」；或点地址栏输入 `powershell` 回车。
3. 命令行开头显示的路径末尾应是 `agentic-rag-cn`，说明位置对了。

# 第③步：填入 API Key
1. 进入 `project` 文件夹，用**记事本**打开 `.env`（看不到就在资源管理器「查看」里勾「隐藏的项目」）。
2. 把这行等号后面换成你的 Key（等号两边别留空格，别加引号）：
   ```
   API_KEY=在这里粘贴你的Key
   ```
3. `MODEL_ID` 保持 `glm-4.7`（想更快可改 `glm-4-flash`）。Ctrl+S 保存。

# 第④步：创建并激活虚拟环境（用 3.12 建！）
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
python --version
```
- **第三行 `python --version` 必须显示 `Python 3.12.x`**——这是本项目最容易翻车的检查点。若还是 3.9，说明第 0.2 步没装成功，回头重装。
- 若激活报「禁止运行脚本」，先跑 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 再激活。
- 成功后命令行最前面会出现绿色 `(venv)`。**以后每次跑都要先激活到有 (venv)。**
- 若提示 `py` 不是命令：把 `py -3.12` 换成 3.12 的完整路径，通常是
  `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv venv`

# 第⑤步：安装依赖
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- 第一次要几分钟，滚进度条正常。已改用智谱嵌入，**不用下载 torch**。
- 看到 `Successfully installed ...` 即装完。中断了就重跑这条命令（已装的会跳过）。

# 第⑥步：启动
```powershell
python project/app.py
```
出现 `Running on local URL:  http://127.0.0.1:7860` 即成功，浏览器打开 **http://localhost:7860**。
⚠️ 这个窗口不要关，关了服务就停。

> 💡 想在启动前先逐个自检每个文件的语法（附真实终端截图），见 **《运行与自检指南.md》**。

# 第⑦步：使用
1. 「文档管理」→ 拖入中文 PDF → 点「添加文档」→ 等索引完成。
2. 「对话」→ 用中文提问。第一次回答可能等 40~70 秒（模型带推理、一次问答要调好几次），属正常。

---

# 以后再次启动（不用重装）
1. 在 `agentic-rag-cn` 里打开 PowerShell。
2. `venv\Scripts\Activate.ps1`（看到 `(venv)`）。
3. `python project/app.py`。
之前上传过的文档还在，不用重传。

# 停止
运行窗口里按 `Ctrl + C`，或直接关窗口。

---

# 常见问题

| 现象 | 解决办法 |
|---|---|
| 装依赖报 `No matching distribution found for langchain-openai` | Python 版本太老（你是 3.9？）。装 Python 3.11/3.12，删掉旧 venv 用新版本重建，见「准备工作」 |
| `No module named 'gradio'` 等找不到模块 | 依赖没装成功（多半也是 Python 版本太老或没激活 venv）。先确认 `python --version` 是 3.11/3.12 且命令行有 `(venv)`，再装依赖 |
| `python` 不是内部或外部命令 | 装 Python 时没勾 "Add to PATH"，重装并勾上 |
| 激活报「禁止运行脚本」 | 先跑 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 再激活 |
| 提示未检测到 API Key / 401 | 检查 `project/.env` 里 `API_KEY=` 填了没、有没有空格或引号 |
| 报 `model not found` | 把 `.env` 里 `MODEL_ID` 换成 `glm-4.6` |
| 回答太慢 | `.env` 里 `MODEL_ID` 换成 `glm-4-flash` 或 `glm-4-air` |
| 报 `code 1210 参数有误` | 若你自己改了模型/参数：智谱不支持强制指定 tool_choice，结构化输出要用 json 模式 |
| 检索总是查不到 | `config.py` 里把 `RETRIEVAL_SCORE_THRESHOLD` 改成 `0`，重启 |
| 改配置后结果乱/维度不匹配 | 删掉根目录 `qdrant_db`、`parent_store`、`markdown_docs` 三个文件夹，重启后重新上传 |
| 想看 Agent 每一步 | `config.py` 里把 `EXECUTION_LOGGING_ENABLED` 改成 `True` |

---

# 目录结构
```
agentic-rag-cn/
├── requirements.txt
├── README_中文.md
├── LICENSE
├── project/
│   ├── .env                ← 填 API Key
│   ├── app.py              启动入口
│   ├── config.py           全局配置
│   ├── document_chunker.py 切块器
│   ├── utils.py            工具函数
│   ├── rag_agent/          智能体：graph/nodes/edges/tools/prompts/schemas/graph_state
│   ├── db/                 向量库 + 父块存储
│   ├── core/               系统装配、文档处理、聊天流、日志、可观测性
│   └── ui/                 Gradio 界面 + 样式
└── （运行后自动生成 qdrant_db / parent_store / markdown_docs）
```
