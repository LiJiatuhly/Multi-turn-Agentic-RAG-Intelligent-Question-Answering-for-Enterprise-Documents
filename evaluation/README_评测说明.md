# RAGAS 评测接入说明

把 `evaluation/` 整个目录放到仓库根目录，与 `project/` 平级：

```
仓库根/
├── project/            # 你现有的代码，一行都不用改
├── markdown_docs/      # 语料
└── evaluation/         # 本目录
    ├── eval_config.py          # 统一配置（两个环境共用，零依赖）
    ├── zhipu_embed.py          # 智谱 embedding 的独立实现（评测环境用）
    ├── gen_testset.py          # ① 语料 → 中文测试集
    ├── run_rag.py              # ② 测试集 → 跑图采集
    ├── score.py                # ③ 采集结果 → RAGAS 打分
    ├── requirements-eval.txt
    └── artifacts/              # 所有产物都落在这（自动创建）
```

---

## ⚠️ 为什么必须开两个虚拟环境

ragas 0.4.3 的 `llms/base.py` 第 12 行有一句硬 import：

```python
from langchain_community.chat_models.vertexai import ChatVertexAI
```

这个模块在 langchain-community 0.4（LangChain 1.x 线）里已经被移除了。也就是说
**ragas 0.4.3 把你钉死在 langchain-core 0.3.x**，而 LangGraph 1.x 要求
`langchain-core>=1.4`。两者在同一个 venv 里装不下——实测报错：

```
langgraph 1.2.10 requires langchain-core<2,>=1.4.7, but you have langchain-core 0.3.86
```

所以方案是**把"跑图"和"打分"拆到两个环境**，中间用 `artifacts/runs.jsonl` 这个文件对接。
这本来也是更好的架构：改指标不用重跑图，跑图挂了能断点续跑。

```bash
# 评测环境（新建）
python -m venv .venv-eval
.venv-eval\Scripts\activate          # Windows
pip install -r evaluation/requirements-eval.txt

# 项目环境：就是你现在跑 app.py 的那个，什么都不用装
```

---

## 运行顺序

### ① 生成测试集 —— 评测环境

```bash
python evaluation/gen_testset.py --size 5      # 先小样跑通
python evaluation/gen_testset.py --size 30     # 正式生成
```

RAGAS 会先把 `markdown_docs/` 建成知识图谱（LLM 抽标题、摘要、实体、主题，再用向量建边），
然后按三种出题器造题，产出 `问题 + 标准答案 + 来源原文`。

- 知识图谱缓存在 `artifacts/knowledge_graph.json`，第二次跑直接复用；语料变了才加 `--rebuild-kg`
- 默认会把出题 prompt 适配成中文（`adapt_prompts("chinese")`），不做这步中文题面会很别扭
- **跑完请打开 `artifacts/testset_preview.csv` 人工过一遍**，把不通顺、答非所问的题删掉。
  自动生成的题里总有几道是坏的，不筛就直接评，等于拿噪声当基准

### ② 跑图采集 —— 项目环境

```bash
python evaluation/run_rag.py --limit 3    # 先验证链路
python evaluation/run_rag.py              # 全跑（已完成的自动跳过）
```

**跑之前把 Gradio 界面关掉**——本地 Qdrant 是文件锁，两个进程会抢 `qdrant_db/`。

这一步不 import ragas，只调你的 `RAGSystem`。每题一个全新 `thread_id`，避免上一题的对话摘要污染下一题。
中途 Ctrl+C 也没关系，已完成的都已落盘，重跑自动续上。

`runs.jsonl` 遵循 JSONL 规范，每条记录必须占一整行，适合程序读取但不适合直接阅读。
脚本会同时生成 `artifacts/runs_preview.md`，按“问题 / RAG 回答 / 标准答案 / 来源”排版，
人工检查回答时请看这个 Markdown 文件；Ragas 评分仍读取 `runs.jsonl`。

`search_child_chunks` 现在是“混合召回候选 → cross-encoder rerank → 返回 top-k”，
所以 `runs.jsonl` 采集到的 contexts 已经是重排后的顺序；无需在评测脚本里重复排序。

### ③ 打分 —— 评测环境

```bash
python evaluation/score.py            # 两个口径都评
python evaluation/score.py --fast     # 兼容旧命令；整轮级仍评固定四指标
```

产出 `artifacts/scores_turn.csv`、`scores_sub.csv`、`report.md`。

为控制裁判延迟，`score.py` 默认按 reranker 顺序最多发送前 8 个检索块、总计 5000 字；
原始完整检索结果仍保存在 `runs.jsonl`，不会被覆盖。该限制可通过
`EVAL_JUDGE_MAX_CONTEXTS` 和 `EVAL_JUDGE_MAX_CONTEXT_CHARS` 调整。

---

## 两个评测口径

`rewrite_query` 会把一个问题拆成 1~3 个子问题并行跑，所以一次提问会产生两层数据：

| 口径 | 样本 | 指标 | 用来回答 |
| --- | --- | --- | --- |
| **整轮级** | 原始问题 + `aggregate_answers` 的最终回答 + 合并去重后的原文 | context_recall、context_precision、faithfulness、answer_relevancy | 用户实际体验到的效果 |
| **子问题级** | 每个并行 Agent 各自的子问题 / 答案 / 原文 | faithfulness、answer_relevancy、context_precision（无参考版） | 并行的哪一路检索拖了后腿 |

子问题级**没有参考答案**——测试集里的 `reference` 是对着原始问题给的，
拆出来的子问题没有对应的标准答案。所以子问题级只上无参考指标，
硬套 `context_recall` 会得到一堆没意义的分数。

---

## 几个已经处理掉的坑

**父块子块混在一起。** `retrieved_contexts` 里既有 500 字的子块也有 2000~4000 字的父块，
父块通常已经包含了子块，同一段原文被算两次会稀释 context precision。
`run_rag.py` 的 `clean_contexts()` 做了包含关系去重，只保留信息最全的那块
（开关：`eval_config.DROP_CHILD_COVERED_BY_PARENT`）。

**澄清中断。** 图是 `interrupt_before=["request_clarification"]` 编译的，
问题被判"不清晰"就停住等人补充，批量评测没人在旁边。
默认自动补一轮通用回复再继续，仍然停住就记为 `clarification` 跳过，不计入分数
（开关：`AUTO_CLARIFY_ROUNDS`，设 0 = 一律跳过）。

**裁判模型的自我偏好。** 默认拿 `MODEL_ID`（glm-4.7）给它自己打分，分数会偏高。
对外汇报前在 `.env` 里加一个不同的 `JUDGE_MODEL_ID`，哪怕是同厂另一个档位也好。

**速率限制。** 默认 `JUDGE_MAX_WORKERS=4`。报 429 就往下调。

---

## 关于版本：为什么选 0.4.3 而不是 0.3.x

0.4 把指标搬到了 `ragas.metrics.collections`，并且推 `@experiment()` 取代 `evaluate()`。
但源码里 `evaluate()` 有一句 `isinstance(m, Metric)` 校验，
**新版 collections 指标塞进 `evaluate()` 会直接 TypeError**，两套 API 不能混。

这里选的是 0.4.3 上的稳妥组合：

- LLM 用 **新的** `llm_factory(model, client=OpenAI(base_url=...))` ——
  没被废弃，而且对 OpenAI 兼容接口最友好（GLM 若报 `response_format` 错，
  在 `.env` 里把 `INSTRUCTOR_MODE` 改成 `md_json` 即可）
- 指标和 `evaluate()` 走 **老路径** —— 老指标 + instructor LLM 是 ragas 自己内部的默认组合，
  能跑、有 `.to_pandas()` 表格。只是会打 DeprecationWarning，脚本里已静音
- Embedding 用 `LangchainEmbeddingsWrapper` 包你那份 `ZhipuEmbeddings` ——
  它是目前唯一同时满足「出题 transforms 要 `embed_text`」和
  「ResponseRelevancy 要 `embed_query`」的类（新版 `OpenAIEmbeddings` 只有 `embed_text`/`embed_texts`）

等 ragas 出到 1.0、`evaluate()` 真的被移除时再整体迁到 `@experiment()`，
到时候 `runs.jsonl` 这份采集结果还能直接复用——这也是把两步拆开的好处之一。
