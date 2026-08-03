# Agentic RAG 中文文档问答系统说明书

这份文档是本项目的学习入口。建议先读“整体理解”，再按“启动项目”跑起来，最后回到“源码阅读顺序”逐个看文件。

## 1. 项目解决什么问题

系统把 PDF 或 Markdown 文档切成可检索的文本块，用户在 Gradio 页面提问后，Agent 会：

1. 理解和改写用户问题；
2. 必要时把复杂问题拆成 1~3 个子问题；
3. 对每个子问题进行混合检索；
4. 让模型判断是否需要继续检索；
5. 根据检索证据生成答案；
6. 汇总并返回最终回复。

它不是一次“向量搜索后直接问模型”的简单 RAG，而是由 LangGraph 控制的检索循环。

## 2. 目录结构

```text
agentic-rag-cn/
├── README.md                         项目简介和特性速览
├── RAG_SYSTEM_GUIDE.md               本说明书
├── requirements.txt                  项目运行环境依赖
├── LICENSE                           开源许可证
├── project/                          主应用源码
│   ├── app.py                        程序启动入口
│   ├── config.py                     全局配置
│   ├── utils.py                      PDF 转 Markdown、文本和 token 工具
│   ├── document_chunker.py            父子分块
│   ├── core/                         应用服务层
│   ├── db/                           Qdrant 和父块存储
│   ├── rag_agent/                    LangGraph Agent
│   ├── ui/                           Gradio 界面
│   └── assets/                       页面资源
├── markdown_docs/                    已转换的 Markdown 语料
├── parent_store/                     父块 JSON 数据（运行后生成）
├── qdrant_db/                        本地 Qdrant 数据（运行后生成）
├── evaluation/                       Ragas 评测脚本和产物
└── docs/                             学习资料和面试资料
    ├── learning/                     学习路线、提示词资料
    └── interview_materials/          面试相关文件
```

`venv/` 和 `.venv-eval/` 是本机虚拟环境；不要把它们当作源码阅读。`parent_store/`、`qdrant_db/` 和 `evaluation/artifacts/` 是运行产物。

### 2.1 其他资料文件

这些文件不参与 RAG 运行，只是项目附带资料：

| 文件 | 作用 |
| --- | --- |
| `docs/learning/Agent应用开发工程师-学习路线.md` | Agent 应用开发学习路线。 |
| `docs/learning/开源项目教程-精读学习提示词-v13.1.md` | 阅读和分析开源项目时使用的提示词资料。 |
| `docs/interview_materials/README.md` | 面试材料的构建说明。 |
| `docs/interview_materials/render.py` | 面试文档渲染脚本。 |
| `docs/interview_materials/build.sh` | 面试材料构建命令。 |
| `docs/interview_materials/manual.html` | 已生成的面试材料网页。 |
| `docs/interview_materials/fc1.dot`、`fc2.dot` | 面试材料中的 Graphviz 图源文件。 |
| `docs/interview_materials/*.docx`、`*.pdf` | 面试回答文档成品。 |

## 3. 第一次启动

### 3.1 创建项目环境

在项目根目录执行：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果已经有 `venv/`，直接激活即可。

### 3.2 配置智谱 API

创建 `project/.env`：

```dotenv
API_KEY=你的智谱API_KEY
BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODEL_ID=glm-4.7
```

代码读取的是 `API_KEY`，不是 `ZHIPUAI_API_KEY`。`MODEL_ID` 可以换成你账号可用的 GLM 模型。

### 3.3 启动应用

```powershell
venv\Scripts\Activate.ps1
python project\app.py
```

浏览器打开 `http://127.0.0.1:7860`。

## 4. 文档如何进入知识库

文档入库在 Gradio 的“文档管理”页完成：

1. 上传 PDF 或 Markdown；
2. PDF 被转换到 `markdown_docs/`；
3. `DocumentChunker` 先生成父块，再把父块切成子块；
4. 父块保存到 `parent_store/`；
5. 子块写入本地 Qdrant 的 `document_child_chunks` 集合；
6. 子块带有 `parent_id`，后续可以回取完整父块。

重复上传同名 Markdown 会跳过。上传失败时，文档管理器会删除已写入的父块和 Markdown，避免留下半成品。

## 5. 一个问题的完整流程

```text
用户问题
  │
  ▼
summarize_history       压缩旧对话，保留近期消息
  │
  ▼
rewrite_query            判断问题是否清晰，改写/拆分为 1~3 个子问题
  │
  ├── 不清晰 → interrupt 暂停，等待用户补充
  │
  └── 清晰 → Send 并行启动多个 Agent 子图
                  │
                  ▼
          orchestrator 判断是否需要检索
                  │
                  ▼
          search_child_chunks
            Qdrant 稠密向量 + BM25 混合召回 20 个候选
            cross-encoder reranker 精排，返回 top 7
                  │
                  ├── 证据不足 → 再次检索
                  └── 证据足够 → collect_answer
                  │
                  ▼
          aggregate_answers       合并并行子问题答案
                  │
                  ▼
              返回用户
```

### 5.1 主图和 Agent 子图

主图负责“一整轮问题”：历史、改写、并行分发和汇总。

Agent 子图负责“一个子问题”：模型调用工具、读取工具结果、决定继续检索或回答。

这种双层结构的关键是：主图不需要知道每个 Agent 如何检索，只接收子 Agent 的答案和上下文。

### 5.2 两阶段检索

现有检索流程不是替换，而是在原来的混合检索后增加了精排：

```text
第一阶段：召回 20 个候选
  dense embedding + BM25 + Qdrant HYBRID/RRF

第二阶段：精排并截断
  jinaai/jina-reranker-v2-base-multilingual
  只把最相关的 7 个子块交给 Agent
```

模型第一次真正检索时会下载 reranker。可通过环境变量更换模型：

```dotenv
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
```

如果 reranker 下载或推理失败，工具会回退到原有的混合召回结果，Agent 不会因此直接崩溃。

## 6. 每个源码文件做什么

### 6.1 根级和基础文件

| 文件 | 作用 |
| --- | --- |
| `project/app.py` | 唯一的应用启动入口，加载 `.env`，创建并启动 Gradio。 |
| `project/config.py` | 集中配置模型、路径、Qdrant、分块大小、检索数量、Agent 预算和 reranker。 |
| `project/utils.py` | PDF 转 Markdown、目录清理、token 估算等通用函数。 |
| `project/document_chunker.py` | 按 Markdown 标题和字符长度生成父块、子块。 |

### 6.2 `project/core/`：应用服务层

| 文件 | 作用 |
| --- | --- |
| `core/rag_system.py` | 总装类 `RAGSystem`，创建向量库、模型、工具和 Agent 图。 |
| `core/document_manager.py` | 处理上传、转换、切块、入库和失败回滚。 |
| `core/chat_interface.py` | 把 LangGraph 流式消息转换成 Gradio 可显示的消息。 |
| `core/execution_logger.py` | 调试日志，记录节点、路由、工具和错误。 |
| `core/observability.py` | 可选的 Langfuse 可观测性接入。 |
| `core/__init__.py` | Python 包标记文件。 |

### 6.3 `project/db/`：数据存储

| 文件 | 作用 |
| --- | --- |
| `db/vector_db_manager.py` | 创建 Qdrant 本地客户端；提供智谱稠密 embedding 和 FastEmbed BM25；配置 HYBRID 检索。 |
| `db/parent_store_manager.py` | 保存、读取、删除父块 JSON。 |
| `db/__init__.py` | Python 包标记文件。 |

### 6.4 `project/rag_agent/`：Agent 核心

| 文件 | 作用 |
| --- | --- |
| `rag_agent/graph.py` | 编译 Agent 子图和主图，注册节点、工具和边。 |
| `rag_agent/graph_state.py` | 定义主图 `State`、子图 `AgentState` 以及列表/集合 reducer。 |
| `rag_agent/nodes.py` | 所有节点逻辑：摘要、改写、编排、压缩、fallback、收集和汇总。 |
| `rag_agent/edges.py` | 条件路由：问题是否清晰、Agent 是继续工具调用还是回答。 |
| `rag_agent/tools.py` | Agent 可调用的两个工具：搜索子块、回取父块；这里接入 reranker。 |
| `rag_agent/reranker.py` | 对混合召回候选做 cross-encoder 精排。 |
| `rag_agent/prompts.py` | 摘要、改写、编排、压缩、兜底和汇总提示词。 |
| `rag_agent/schemas.py` | `QueryAnalysis` 等 Pydantic 结构化输出模型。 |
| `rag_agent/数据流转对照表.md` | 状态字段的写入者、读取者和 reducer 对照表。 |
| `rag_agent/__init__.py` | Python 包标记文件。 |

### 6.5 `project/ui/`：界面

| 文件 | 作用 |
| --- | --- |
| `ui/gradio_app.py` | 创建文档管理页和对话页，绑定 `DocumentManager`、`ChatInterface`。 |
| `ui/css.py` | 页面 CSS。 |
| `ui/__init__.py` | Python 包标记文件。 |
| `assets/chatbot_avatar.png` | 聊天头像资源。 |

### 6.6 `evaluation/`：Ragas 评测

| 文件 | 作用 |
| --- | --- |
| `evaluation/eval_config.py` | 两个环境共用的评测路径、模型和参数。 |
| `evaluation/gen_testset.py` | 用 Ragas 从 Markdown 语料生成问题、标准答案和参考上下文。 |
| `evaluation/curate_testset.py` | 把人工审核后的 CSV 转成正式测试集 JSONL。 |
| `evaluation/run_rag.py` | 在项目环境中逐题运行真实 Agent，保存答案和检索上下文。 |
| `evaluation/score.py` | 在评测环境中运行 Ragas 指标并生成 CSV/Markdown 报告。 |
| `evaluation/zhipu_embed.py` | 评测环境使用的智谱 embedding 轻量封装。 |
| `evaluation/eval_utils.py` | 测试集校验、语料指纹、原子写入等公共工具。 |
| `evaluation/validate_eval.py` | 不调用模型，检查测试集、manifest 和运行记录。 |
| `evaluation/test_eval.py` | 评测公共工具的离线单元测试。 |
| `evaluation/requirements-eval.txt` | Ragas 独立环境依赖。 |
| `evaluation/README_评测说明.md` | 评测设计、两个虚拟环境和指标说明。 |

## 7. Ragas 评测怎么跑

Ragas `0.4.3` 与项目的 LangGraph 1.x 依赖冲突，因此必须使用两个环境。

### 7.1 创建评测环境

```powershell
python -m venv .venv-eval
.venv-eval\Scripts\Activate.ps1
pip install -r evaluation\requirements-eval.txt
```

### 7.2 生成并审核测试集

关闭 Gradio 后，在评测环境执行：

```powershell
.venv-eval\Scripts\Activate.ps1
python evaluation\gen_testset.py --size 5
```

打开 `evaluation/artifacts/testset_preview.csv`，在“保留”列填写 `是` 或 `否`，然后执行：

```powershell
python evaluation\curate_testset.py
```

### 7.3 跑真实 Agent

切换到项目环境，并确保 Gradio 已关闭，因为本地 Qdrant 需要独占文件锁：

```powershell
venv\Scripts\Activate.ps1
python evaluation\run_rag.py
```

结果写入 `evaluation/artifacts/runs.jsonl`。

### 7.4 评分

```powershell
.venv-eval\Scripts\Activate.ps1
python evaluation\score.py --fast
```

主要产物：

- `scores_turn.csv`：整轮问题的检索和回答评分；
- `scores_sub.csv`：子问题级评分；
- `scores_agent.csv`：迭代次数、工具调用次数、搜索次数等行为数据；
- `report.md`：汇总报告。

## 8. 配置中最值得先理解的参数

| 参数 | 含义 |
| --- | --- |
| `DENSE_MODEL` | 智谱稠密 embedding 模型。 |
| `SPARSE_MODEL` | FastEmbed BM25 模型。 |
| `RETRIEVAL_CANDIDATE_K` | rerank 前召回多少候选，默认 20。 |
| `DEFAULT_RETRIEVAL_K` | 最终返回多少子块，默认 7。 |
| `RERANKER_MODEL` | cross-encoder 模型。 |
| `MAX_TOOL_CALLS` | 一个子 Agent 最多调用多少次工具。 |
| `MAX_ITERATIONS` | 一个子 Agent 最多循环多少轮。 |
| `BASE_TOKEN_THRESHOLD` | 触发上下文压缩的基础阈值。 |
| `MAIN_HISTORY_MESSAGES_TO_KEEP` | 主图保留的近期对话消息数。 |

建议学习时一次只改一个参数，然后用相同问题比较答案和检索上下文。

## 9. 推荐源码阅读顺序

1. 先看本文件第 5 节，建立完整数据流；
2. 看 `project/core/rag_system.py`，理解系统如何组装；
3. 看 `project/rag_agent/graph.py`，理解两张图如何连接；
4. 看 `project/rag_agent/graph_state.py`，理解状态和 reducer；
5. 看 `project/rag_agent/edges.py`，理解路由；
6. 看 `project/rag_agent/nodes.py` 中 `rewrite_query`、`orchestrator`、`collect_answer`、`aggregate_answers`；
7. 看 `project/rag_agent/tools.py` 和 `reranker.py`，理解检索；
8. 最后看 `document_chunker.py`、`vector_db_manager.py` 和 `parent_store_manager.py`，理解数据从哪里来、存在哪里；
9. 最后再看 `evaluation/`，把 Ragas 当成系统外部的实验工具。

## 10. 常见问题

### 没有 API Key

检查 `project/.env` 中是否是 `API_KEY=...`，并确认从项目根目录启动。

### Qdrant 文件被占用

关闭 Gradio 和其他 `run_rag.py` 进程，一次只保留一个访问 `qdrant_db/` 的进程。

### 修改了 embedding 模型后报维度不匹配

旧 Qdrant 集合的维度和新模型不同。确认不再需要旧库后，删除 `qdrant_db/`，重新启动并重新上传文档。

### reranker 首次运行很慢

首次运行需要从 Hugging Face 下载模型；下载完成后会使用本地缓存。之后检索阶段只进行本地 ONNX 推理。

### 修改文档后评测结果没有变化

重新生成测试集时使用 `--rebuild-kg`，并重新运行 `run_rag.py`。评测脚本会通过语料指纹拒绝混用旧测试集。
