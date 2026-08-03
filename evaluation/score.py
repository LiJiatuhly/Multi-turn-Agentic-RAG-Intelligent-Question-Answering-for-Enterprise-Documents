"""【评测环境】读 runs.jsonl，用 RAGAS 打分，出 CSV 明细 + Markdown 报告。

两个口径分开评，因为它们要回答的问题不一样：

  整轮级（有标准答案）—— 评"用户最终看到的那个回答"
      context_recall          标准答案里的句子，有多少能被检索到的原文支撑（LLM 判定）
      context_precision       检索回来的原文里，真正有用的排得靠不靠前
      faithfulness            回答有没有超出原文瞎编
      answer_relevancy        回答有没有答在点上

  子问题级（没有标准答案 —— rewrite_query 拆出来的子问题不带 reference）
      faithfulness / answer_relevancy / context_precision(无参考版)
      用来定位：并行的哪一路检索拖了后腿

用法：
    python score.py                 # 两个口径都评
    python score.py --scope turn    # 只评整轮
    python score.py --fast          # 兼容旧命令；整轮级仍评固定四指标

Ragas 的输入不是原始文档，而是每条真实运行记录中的五个字段：
user_input（问题）、response（你的 Agent 回答）、retrieved_contexts（实际检索到的内容）、
reference（标准答案）和 reference_contexts（出题时使用的参考原文）。
因此评分衡量的是“你的 RAG 对真实问题回答得怎么样”，不是单独评价一个 LLM。
"""

import argparse
import csv
import json
import statistics
import sys
import threading
import time
import warnings

# Must be installed before importing deprecated metric aliases.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

import eval_config as C
from eval_utils import file_sha256, read_jsonl, utc_now
from zhipu_embed import ZhipuEmbeddings

# 0.4 里这批指标从 ragas.metrics 挪到了 ragas.metrics.collections，老路径会告警。
# 但 collections 版本的指标【不能塞进 evaluate()】（内部有 isinstance(m, Metric) 校验会直接 TypeError），
# 所以这里仍走 evaluate() + 老指标这条路，把告警静音即可。
def build_judge():
    # 裁判 LLM 负责 faithfulness/context_recall/context_precision 等判断；
    # 裁判 embedding 负责 answer_relevancy 等向量相似度指标。
    client = OpenAI(
        api_key=C.API_KEY,
        base_url=C.BASE_URL,
        timeout=180.0,
        max_retries=C.JUDGE_MAX_RETRIES,
    )
    judge_args = {
        "max_tokens": C.JUDGE_MAX_TOKENS,
        "max_retries": C.JUDGE_MAX_RETRIES,
    }
    if C.JUDGE_DISABLE_THINKING:
        # Ragas 需要结构化结论而不是长推理；关闭思考可减少延迟和输出截断。
        judge_args["extra_body"] = {"thinking": {"type": "disabled"}}
    llm = llm_factory(C.JUDGE_MODEL_ID, client=client, **judge_args)
    embeddings = LangchainEmbeddingsWrapper(
        ZhipuEmbeddings(
            model=C.EMBEDDING_MODEL,
            api_key=C.API_KEY,
            base_url=C.BASE_URL,
            batch_size=C.EMBEDDING_BATCH_SIZE,
        )
    )
    return llm, embeddings


def load_runs():
    # 只取与本次 run_manifest 测试集哈希匹配的最新记录，避免混入旧实验结果。
    if not C.RUNS_PATH.exists():
        raise SystemExit(f"找不到 {C.RUNS_PATH}，先在项目环境跑 run_rag.py")
    if not C.RUN_MANIFEST.exists():
        raise SystemExit(f"找不到 {C.RUN_MANIFEST}，请重新运行 run_rag.py")
    run_manifest = json.loads(C.RUN_MANIFEST.read_text(encoding="utf-8"))
    testset_hash = run_manifest.get("testset_sha256", "")
    if not C.TESTSET_CURATED_PATH.exists():
        raise SystemExit("找不到正式测试集，请先生成并审核测试集。")
    if file_sha256(C.TESTSET_CURATED_PATH) != testset_hash:
        raise SystemExit("测试集已经变化，runs.jsonl 还是旧结果；请先在项目环境重新运行 run_rag.py。")
    records = [r for r in read_jsonl(C.RUNS_PATH) if r.get("testset_sha256") == testset_hash]
    latest = {}
    for record in records:
        latest[record.get("id")] = record
    ok = [r for r in latest.values() if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["id"])
    skipped = len(latest) - len(ok)
    if skipped:
        print(f"⚠️ 跳过 {skipped} 条非成功记录（澄清中断 / 报错）")
    if not ok:
        raise SystemExit("没有可评的记录。")
    return ok, run_manifest


def write_agent_scores(runs):
    fields = [
        "id", "sub_id", "question", "elapsed_s", "clarify_rounds", "context_count",
        "iteration_count", "tool_call_count", "search_calls", "parent_fetches", "used_fallback",
    ]
    rows = []
    for run in runs:
        for sub in run.get("sub", []):
            keys = sub.get("retrieval_keys", [])
            rows.append({
                "id": run["id"],
                "sub_id": f"{run['id']}#{sub.get('index')}",
                "question": sub.get("question", ""),
                "elapsed_s": run.get("elapsed_s", 0),
                "clarify_rounds": run.get("clarify_rounds", 0),
                "context_count": len(sub.get("contexts", [])),
                "iteration_count": sub.get("iteration_count", 0),
                "tool_call_count": sub.get("tool_call_count", 0),
                "search_calls": sum(1 for key in keys if key.startswith("search::")),
                "parent_fetches": sum(1 for key in keys if key.startswith("parent::")),
                "used_fallback": bool(sub.get("used_fallback", False)),
            })
    with C.SCORES_AGENT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def behavior_report(runs, rows):
    def mean(field):
        values = [float(row[field]) for row in rows]
        return statistics.fmean(values) if values else 0.0

    fallback_count = sum(bool(row["used_fallback"]) for row in rows)
    clarified = sum(int(run.get("clarify_rounds", 0)) > 0 for run in runs)
    return (
        "\n## Agent 行为\n\n"
        "| 指标 | 结果 |\n| --- | ---: |\n"
        f"| 有效整轮 | {len(runs)} |\n"
        f"| 子 Agent 数 | {len(rows)} |\n"
        f"| 平均整轮耗时 | {statistics.fmean(float(r.get('elapsed_s', 0)) for r in runs):.2f}s |\n"
        f"| 平均每个子 Agent 迭代数 | {mean('iteration_count'):.2f} |\n"
        f"| 平均每个子 Agent 工具调用数 | {mean('tool_call_count'):.2f} |\n"
        f"| 平均搜索次数 | {mean('search_calls'):.2f} |\n"
        f"| 平均父块回取次数 | {mean('parent_fetches'):.2f} |\n"
        f"| 触发澄清的整轮 | {clarified}/{len(runs)} |\n"
        f"| 触发预算 fallback 的子 Agent | {fallback_count}/{len(rows)} |\n\n"
        f"行为明细：`{C.SCORES_AGENT.name}`\n"
    )


def compact_contexts(contexts):
    """按检索顺序限制裁判上下文；不做摘要，避免引入新的模型事实。"""
    kept = []
    used_chars = 0
    for context in contexts or []:
        text = str(context).strip()
        if not text or len(kept) >= C.JUDGE_MAX_CONTEXTS:
            break
        remaining = C.JUDGE_MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            # 至少保留排名第一的上下文；后续块超出总预算时直接停止。
            if not kept:
                kept.append(text[:remaining])
            break
        kept.append(text)
        used_chars += len(text)
    return kept or [""]


def build_turn_dataset(runs):
    # 整轮级数据对应用户实际看到的最终回答，因此有 reference 可做有参考评分。
    rows = []
    for r in runs:
        rows.append({
            "user_input": r["user_input"],
            "retrieved_contexts": compact_contexts(r.get("turn_contexts")),
            "response": r.get("final_answer") or "",
            "reference": r.get("reference") or "",
            "reference_contexts": r.get("reference_contexts") or [""],
        })
    return EvaluationDataset.from_list(rows), [r["id"] for r in runs]


def build_sub_dataset(runs):
    # 子问题是 Agent 内部改写结果，没有一一对应的标准答案，只使用无参考指标。
    rows, labels = [], []
    for r in runs:
        for sub in r.get("sub", []):
            rows.append({
                "user_input": sub.get("question", ""),
                "retrieved_contexts": compact_contexts(sub.get("contexts")),
                "response": sub.get("answer") or "",
            })
            labels.append(f"{r['id']}#{sub.get('index')}")
    return (EvaluationDataset.from_list(rows), labels) if rows else (None, [])


def run_eval(dataset, metrics, llm, embeddings):
    # Ragas 在这里逐条调用指标：有些指标调用裁判 LLM，有些使用 embedding 或字符串比较。
    # raise_exceptions=False 让单条失败变成 NaN，不影响其余样本完成。
    stop_heartbeat = threading.Event()
    rows = dataset.to_list()
    total = len(rows)
    state = {"completed": 0, "current": 1, "started": time.monotonic()}
    output_lock = threading.Lock()

    def show_status(final=False):
        with output_lock:
            if final:
                message = f"[完成] {total}/{total} 条样本"
            else:
                elapsed = int(time.monotonic() - state["started"])
                message = (
                    f"[评分中] 已完成 {state['completed']}/{total} | "
                    f"正在处理 {state['current']}/{total} | 当前等待 {elapsed}s"
                )
            sys.stdout.write(f"\r\033[2K{message}")
            sys.stdout.flush()

    def heartbeat():
        while not stop_heartbeat.wait(C.JUDGE_HEARTBEAT_SECONDS):
            show_status()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    frames = []
    try:
        show_status()
        for index, row in enumerate(rows, 1):
            state.update(current=index, started=time.monotonic())
            result = evaluate(
                dataset=EvaluationDataset.from_list([row]),
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(
                    max_workers=C.JUDGE_MAX_WORKERS,
                    timeout=C.JUDGE_TIMEOUT,
                    max_retries=C.JUDGE_MAX_RETRIES,
                    max_wait=C.JUDGE_MAX_WAIT,
                ),
                raise_exceptions=False,  # 单条挂了记 NaN，不让整批白跑
                show_progress=False,     # 使用本文件的单行进度，避免两套进度互相覆盖
            )
            frames.append(result.to_pandas())
            state["completed"] = index
            show_status()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    finally:
        stop_heartbeat.set()
        thread.join(timeout=1)
        show_status(final=state["completed"] == total)
        print()


def to_frame(result, labels, id_col):
    df = result.copy()
    df.insert(0, id_col, labels[: len(df)])
    return df


def summarize(df, id_col):
    """算每个指标的均值 / 中位数 / 有效条数。"""
    skip = {id_col, "user_input", "retrieved_contexts", "response", "reference", "reference_contexts"}
    lines = []
    for col in df.columns:
        if col in skip:
            continue
        series = df[col]
        if series.dtype.kind not in "fi":
            continue
        valid = series.dropna()
        if valid.empty:
            lines.append((col, None, None, 0, len(series)))
        else:
            lines.append((col, valid.mean(), valid.median(), len(valid), len(series)))
    return lines


def md_table(summary):
    out = ["| 指标 | 均值 | 中位数 | 有效/总数 |", "| --- | --- | --- | --- |"]
    for name, mean, median, valid, total in summary:
        if mean is None:
            out.append(f"| {name} | — | — | 0/{total} |")
        else:
            out.append(f"| {name} | {mean:.3f} | {median:.3f} | {valid}/{total} |")
    return "\n".join(out)


def worst_cases(df, id_col, n=5):
    skip = {id_col, "user_input", "retrieved_contexts", "response", "reference", "reference_contexts"}
    numeric = [c for c in df.columns if c not in skip and df[c].dtype.kind in "fi"]
    if not numeric:
        return ""
    tmp = df.copy()
    tmp["_avg"] = tmp[numeric].mean(axis=1, skipna=True)
    worst = tmp.nsmallest(n, "_avg")
    out = ["| id | 均分 | 问题 |", "| --- | --- | --- |"]
    for _, row in worst.iterrows():
        q = str(row.get("user_input", ""))[:60].replace("|", "／").replace("\n", " ")
        out.append(f"| {row[id_col]} | {row['_avg']:.3f} | {q} |")
    return "\n".join(out)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["turn", "sub", "both"], default="both")
    parser.add_argument("--fast", action="store_true", help="兼容旧命令；整轮级仍固定评四个核心指标")
    parser.add_argument("--behavior-only", action="store_true", help="只生成 Agent 行为报告，不调用裁判模型")
    args = parser.parse_args()

    if not C.API_KEY and not args.behavior_only:
        raise SystemExit("没读到 API_KEY，请检查 project/.env")

    runs, run_manifest = load_runs()
    agent_rows = write_agent_scores(runs)
    print(f"待评 {len(runs)} 条；裁判模型 {C.JUDGE_MODEL_ID}")
    if C.JUDGE_MODEL_ID == C.MODEL_ID:
        print("⚠️ 裁判模型与被测模型是同一个，分数会偏高（自我偏好）。"
              "对外汇报前建议在 .env 里配一个不同的 JUDGE_MODEL_ID。")

    report = [
        "# RAGAS 评测报告\n",
        f"生成时间：`{utc_now()}`　样本：{len(runs)} 条有效记录　裁判模型：`{C.JUDGE_MODEL_ID}`\n",
        f"测试集：`{run_manifest.get('testset_sha256', '')[:12]}`　"
        f"语料：`{run_manifest.get('corpus', {}).get('sha256', '')[:12]}`\n",
        behavior_report(runs, agent_rows),
    ]

    if args.behavior_only:
        C.REPORT_PATH.write_text("".join(report), encoding="utf-8")
        print(f"Agent 行为报告已生成：{C.REPORT_PATH}")
        return

    llm, embeddings = build_judge()

    if args.scope in ("turn", "both"):
        # 整轮级固定评价四个核心指标：检索质量两项 + 回答质量两项。
        turn_metrics = [
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(strictness=C.ANSWER_RELEVANCY_STRICTNESS),
            LLMContextPrecisionWithReference(),
        ]
        print("\n===== 整轮级评测 =====")
        dataset, labels = build_turn_dataset(runs)
        result = run_eval(dataset, turn_metrics, llm, embeddings)
        df = to_frame(result, labels, "id")
        df.to_csv(C.SCORES_TURN, index=False, encoding="utf-8-sig")
        summary = summarize(df, "id")
        print(md_table(summary))
        report += ["\n## 整轮级（用户实际看到的最终回答）\n", md_table(summary),
                   "\n\n### 得分最低的几条\n", worst_cases(df, "id"), f"\n\n明细：`{C.SCORES_TURN.name}`\n"]

    if args.scope in ("sub", "both"):
        # 子问题级：定位并行 Agent 中是哪一路检索或回答质量较差。
        dataset, labels = build_sub_dataset(runs)
        if dataset is None:
            print("没有子问题记录，跳过子问题级评测。")
        else:
            sub_metrics = [
                Faithfulness(),
                ResponseRelevancy(strictness=C.ANSWER_RELEVANCY_STRICTNESS),
            ]
            sub_metrics.append(LLMContextPrecisionWithoutReference())
            print(f"\n===== 子问题级评测（{len(labels)} 条）=====")
            result = run_eval(dataset, sub_metrics, llm, embeddings)
            df = to_frame(result, labels, "sub_id")
            df.to_csv(C.SCORES_SUB, index=False, encoding="utf-8-sig")
            summary = summarize(df, "sub_id")
            print(md_table(summary))
            report += ["\n## 子问题级（并行 Agent 各自的那一路）\n", md_table(summary),
                       "\n\n### 得分最低的几条\n", worst_cases(df, "sub_id"), f"\n\n明细：`{C.SCORES_SUB.name}`\n"]

    report.append(
        "\n## 指标怎么读\n\n"
        "- `context_recall`：标准答案里的每句话能否被检索到的原文支撑（LLM 判定）。"
        "它不是 IR 的召回率，别混着汇报。\n"
        "- `context_precision`：检索回来的块里有用的是否排在前面。低 = 召回噪声多。\n"
        "- `faithfulness`：回答里的每个论断能否在原文中找到依据。低 = 模型在编。\n"
        "- `answer_relevancy`：回答与问题的贴合度（用嵌入算的）。低 = 答非所问或答得太散。\n"
        "**排查顺序**：context_recall 低 → 是检索问题（切块/阈值/k/改写）；"
        "context_recall 高但 faithfulness 低 → 是生成问题（提示词/上下文压缩丢了东西）。\n"
    )

    C.REPORT_PATH.write_text("".join(report), encoding="utf-8")
    print(f"\n报告已生成：{C.REPORT_PATH}")


if __name__ == "__main__":
    main()
