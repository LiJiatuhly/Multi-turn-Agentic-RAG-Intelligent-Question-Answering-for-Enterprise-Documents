"""【项目环境】拿 testset.jsonl 跑你的图，把结果采集成 runs.jsonl。

这一步【不 import ragas】——它跑在你现有的项目环境里，只依赖 project/ 的代码。
评分是下一步（score.py）在另一个环境里做的，两边靠 runs.jsonl 这个文件对接。
好处：改指标不用重跑图；跑图挂了也能断点续跑。

⚠️ 跑之前先把 Gradio 界面关掉 —— 本地 Qdrant 是文件锁，两个进程会抢 qdrant_db/。

用法：
    python run_rag.py                # 断点续跑（已完成的题自动跳过）
    python run_rag.py --limit 3      # 先跑 3 题验证链路
   python run_rag.py --restart      # 清空 runs.jsonl 重跑

这里是评测和你的 RAG 系统的连接点：本文件不导入 Ragas，而是直接调用
project/core/rag_system.py 的 RAGSystem。每道题得到真实 final_answer、检索
上下文和 Agent 行为轨迹，统一写入 runs.jsonl；下一步 score.py 再读取它评分。
"""

import argparse
import json
import sys
import time
import traceback
import uuid

import eval_config as C   # ⚠️ 必须最先 import：它负责把 .env 灌进 os.environ
from eval_utils import (
    corpus_fingerprint,
    file_sha256,
    package_versions,
    read_jsonl,
    utc_now,
    validate_testset,
    write_json_atomic,
)

sys.path.insert(0, str(C.PROJECT_DIR))   # 让 `import config`、`from core...` 能找到模块

from langchain_core.messages import HumanMessage   # noqa: E402
from core.rag_system import RAGSystem              # noqa: E402
import config as project_config                    # noqa: E402


# ============================================================
# --- 检索原文的解析与清洗 ---
# ============================================================
def parse_context(raw: str) -> dict:
    """把工具返回的一块原文拆成结构化字段。

    tools.py 里两个工具吐出来的格式都是这三行：
        Parent ID: xxx
        File Name: yyy
        Content: zzz……（可能很长、含换行）
    """
    parent_id, source, content = "", "", raw
    lines = raw.split("\n")
    body_start = 0
    for i, line in enumerate(lines[:3]):
        if line.startswith("Parent ID:"):
            parent_id = line[len("Parent ID:"):].strip()
            body_start = i + 1
        elif line.startswith("File Name:"):
            source = line[len("File Name:"):].strip()
            body_start = i + 1
        elif line.startswith("Content:"):
            lines[i] = line[len("Content:"):].lstrip()
            body_start = i
            break
    if parent_id or source:
        content = "\n".join(lines[body_start:]).strip()
    return {"parent_id": parent_id, "source": source, "content": content, "raw": raw}


def _flat(text: str) -> str:
    """压掉所有空白，用来做"包含关系"判断。"""
    return "".join(text.split())


def clean_contexts(raw_contexts: list) -> tuple:
    """清洗一批原文块，返回 (给 RAGAS 的纯文本列表, 元信息列表)。

    要解决的问题：retrieved_contexts 里【父块和子块是混在一起的】——
    子块是父块的一部分，同一段原文会被算两次，
    context precision 这类"检索到的有多少是有用的"指标会被稀释。
    """
    # RAG 工具返回的是带 Parent ID/File Name/Content 头的字符串，先解析成结构化块。
    parsed = [parse_context(c) for c in raw_contexts if c and c.strip()]

    # 1) 完全相同的内容去重
    seen, unique = set(), []
    for item in parsed:
        key = _flat(item["content"])
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    # 2) 子块被父块包含 → 丢掉子块，只留信息量最全的那块
    kept = unique
    if C.DROP_CHILD_COVERED_BY_PARENT:
        by_len = sorted(unique, key=lambda x: len(x["content"]), reverse=True)
        kept, kept_flat = [], []
        for item in by_len:
            flat = _flat(item["content"])
            covered = any(flat in bigger for bigger in kept_flat)
            if not covered:
                kept.append(item)
                kept_flat.append(flat)
        # 恢复成原来的召回顺序（顺序对 context precision 有意义）
        order = {id(x): i for i, x in enumerate(unique)}
        kept.sort(key=lambda x: order[id(x)])

    kept = kept[: C.MAX_CONTEXTS_PER_SAMPLE]
    texts = [k["content"] if C.STRIP_CONTEXT_HEADER else k["raw"] for k in kept]
    meta = [{"parent_id": k["parent_id"], "source": k["source"], "chars": len(k["content"])} for k in kept]
    return texts, meta


# ============================================================
# --- 跑一道题 ---
# ============================================================
def run_one(rag: RAGSystem, qid: str, question: str) -> dict:
    """跑一道题，返回一条采集记录。

    关键点：
      · 每题一个全新 thread_id —— 否则上一题的对话摘要会污染这一题
      · 图是 interrupt_before=["request_clarification"] 编译的，
        问题被判"不清晰"会停住，这里按配置自动补充或记为 clarification
    """
    # 每题独立 thread，避免上一题的对话记忆影响下一题，让输入边界固定。
    rag.thread_id = f"eval-{qid}-{uuid.uuid4().hex[:6]}"
    cfg = rag.get_config()

    started = time.time()
    result = rag.agent_graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)

    # --- 处理澄清中断 ---
    clarify_used = 0
    for _ in range(C.AUTO_CLARIFY_ROUNDS):
        state = rag.agent_graph.get_state(cfg)
        if not state.next:
            break
        # 模仿 chat_interface：把补充塞进状态，再用 None 恢复运行
        rag.agent_graph.update_state(cfg, {"messages": [HumanMessage(content=C.AUTO_CLARIFY_REPLY)]})
        result = rag.agent_graph.invoke(None, config=cfg)
        clarify_used += 1

    state = rag.agent_graph.get_state(cfg)
    if state.next:   # 补充完还是停着 → 这题作废
        return {
            "id": qid, "status": "clarification", "clarify_rounds": clarify_used,
            "elapsed_s": round(time.time() - started, 1),
            "stopped_at": list(state.next),
        }

    # --- 提取结果 ---
    messages = result.get("messages", [])
    final_answer = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") == "ai" and content:
            final_answer = content
            break

    subs = []
    all_raw_contexts = []
    # agent_answers 是你的 Agent 图输出的子问题结果；这里把它整理成 Ragas
    # 需要的 user_input/response/retrieved_contexts，同时保留行为统计字段。
    for ans in sorted(result.get("agent_answers", []), key=lambda x: x.get("index", 0)):
        if ans.get("__reset__"):
            continue
        sub_texts, sub_meta = clean_contexts(ans.get("contexts", []))
        subs.append({
            "index": ans.get("index"),
            "question": ans.get("question", ""),
            "answer": ans.get("answer", ""),
            "contexts": sub_texts,
            "context_meta": sub_meta,
            "iteration_count": ans.get("iteration_count", 0),
            "tool_call_count": ans.get("tool_call_count", 0),
            "retrieval_keys": ans.get("retrieval_keys", []),
            "used_fallback": bool(ans.get("used_fallback", False)),
        })
        all_raw_contexts.extend(ans.get("contexts", []))

    turn_texts, turn_meta = clean_contexts(all_raw_contexts)

    if not final_answer.strip():
        raise RuntimeError("图运行结束但没有提取到最终回答")
    if not subs:
        raise RuntimeError("图运行结束但没有采集到任何子 Agent 结果")

    return {
        "id": qid,
        "status": "ok",
        "clarify_rounds": clarify_used,
        "elapsed_s": round(time.time() - started, 1),
        "final_answer": final_answer,
        "rewritten_questions": result.get("rewrittenQuestions", []),
        "turn_contexts": turn_texts,
        "turn_context_meta": turn_meta,
        "sub": subs,
        "agent_behavior": {
            "subproblem_count": len(subs),
            "total_iterations": sum(s["iteration_count"] for s in subs),
            "total_tool_calls": sum(s["tool_call_count"] for s in subs),
            "search_calls": sum(
                1 for s in subs for key in s["retrieval_keys"] if key.startswith("search::")
            ),
            "parent_fetches": sum(
                1 for s in subs for key in s["retrieval_keys"] if key.startswith("parent::")
            ),
            "fallback_count": sum(1 for s in subs if s["used_fallback"]),
            "context_count": len(turn_texts),
        },
    }


# ============================================================
# --- 主流程 ---
# ============================================================
def load_testset(allow_unreviewed: bool):
    if C.TESTSET_CURATED_PATH.exists():
        path = C.TESTSET_CURATED_PATH
        manifest_path = C.CURATED_MANIFEST
    elif allow_unreviewed and C.TESTSET_PATH.exists():
        path = C.TESTSET_PATH
        manifest_path = C.TESTSET_MANIFEST
        print("⚠️ 正在使用未人工确认的候选测试集，仅适合冒烟测试。")
    else:
        raise SystemExit(
            "找不到人工确认测试集。先运行 gen_testset.py，审核 CSV 后运行 curate_testset.py；"
            "冒烟测试可临时加 --allow-unreviewed。"
        )
    if not manifest_path.exists():
        raise SystemExit(f"{path.name} 缺少 manifest，无法保证可复现性：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_corpus = corpus_fingerprint(C.MARKDOWN_DIR)
    if manifest.get("corpus", {}).get("sha256") != current_corpus["sha256"]:
        raise SystemExit("当前语料与测试集生成时不一致，请重新生成测试集。")
    records = validate_testset(read_jsonl(path))
    return records, path, file_sha256(path), current_corpus


def load_done(testset_hash: str) -> set:
    if not C.RUNS_PATH.exists():
        return set()
    done = set()
    with open(C.RUNS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("testset_sha256") != testset_hash:
                    continue
                # Later attempts supersede earlier attempts for the same id.
                if rec.get("status") == "ok":
                    done.add(rec["id"])
                else:
                    done.discard(rec.get("id"))
            except json.JSONDecodeError:
                continue
    return done


def write_runs_preview(testset_hash: str) -> None:
    """把机器读取的 JSONL 转成方便人检查的 Markdown，不参与评分。"""
    if not C.RUNS_PATH.exists():
        return
    latest = {}
    for record in read_jsonl(C.RUNS_PATH):
        if record.get("testset_sha256") == testset_hash:
            latest[record.get("id")] = record

    lines = ["# RAG 真实回答预览\n"]
    for index, record in enumerate(latest.values(), 1):
        lines += [
            f"\n## {index}. {record.get('user_input', '')}\n",
            f"- ID：`{record.get('id', '')}`\n",
            f"- 状态：`{record.get('status', '')}`\n",
            f"- 用时：`{record.get('elapsed_s', 0)}s`\n",
        ]
        if record.get("status") != "ok":
            lines.append(f"- 错误：{record.get('error', record.get('stopped_at', ''))}\n")
            continue

        behavior = record.get("agent_behavior", {})
        lines += [
            f"- 子问题数：`{behavior.get('subproblem_count', 0)}`\n",
            f"- 工具调用数：`{behavior.get('total_tool_calls', 0)}`\n",
            f"- 检索上下文数：`{behavior.get('context_count', 0)}`\n",
            "\n### RAG 回答\n\n",
            record.get("final_answer", "").strip() + "\n",
            "\n### 标准答案\n\n",
            record.get("reference", "").strip() + "\n",
        ]
        sources = sorted({m.get("source", "") for m in record.get("turn_context_meta", []) if m.get("source")})
        if sources:
            lines += ["\n### 检索来源\n\n", *(f"- {source}\n" for source in sources)]

    C.RUNS_PREVIEW.write_text("".join(lines), encoding="utf-8")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全跑）")
    parser.add_argument("--restart", action="store_true", help="清空 runs.jsonl 重跑")
    parser.add_argument("--allow-unreviewed", action="store_true", help="允许使用候选测试集（仅冒烟测试）")
    args = parser.parse_args()

    if args.restart and C.RUNS_PATH.exists():
        C.RUNS_PATH.unlink()

    testset, testset_path, testset_hash, corpus = load_testset(args.allow_unreviewed)
    done = load_done(testset_hash)
    todo = [t for t in testset if t["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"测试集 {len(testset)} 题，已完成 {len(done)} 题，本次待跑 {len(todo)} 题")
    if not todo:
        write_runs_preview(testset_hash)
        print(f"回答预览：{C.RUNS_PREVIEW}")
        return

    print("初始化 RAG 系统（会连一次智谱建向量集合）……")
    rag = RAGSystem()
    rag.initialize()

    invocation_id = f"run-{utc_now().replace(':', '').replace('+00:00', 'Z')}-{uuid.uuid4().hex[:8]}"
    write_json_atomic(C.RUN_MANIFEST, {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "started_at": utc_now(),
        "testset_path": str(testset_path),
        "testset_sha256": testset_hash,
        "corpus": corpus,
        "model": project_config.LLM_MODEL,
        "embedding_model": project_config.DENSE_MODEL,
        "collection": rag.collection_name,
        "retrieval": {
            "mode": "hybrid_rrf",
            "k": project_config.DEFAULT_RETRIEVAL_K,
            "score_threshold": project_config.RETRIEVAL_SCORE_THRESHOLD,
        },
        "agent_budget": {
            "max_iterations": project_config.MAX_ITERATIONS,
            "max_tool_calls": project_config.MAX_TOOL_CALLS,
            "recursion_limit": project_config.GRAPH_RECURSION_LIMIT,
        },
        "versions": package_versions((
            "langgraph", "langchain-core", "langchain-openai", "langchain-qdrant", "qdrant-client"
        )),
    })

    ok = fail = 0
    with open(C.RUNS_PATH, "a", encoding="utf-8") as out:
        for i, item in enumerate(todo, 1):
            qid, question = item["id"], item["user_input"]
            print(f"[{i}/{len(todo)}] {qid} {question[:40]}……", flush=True)
            try:
                record = run_one(rag, qid, question)
            except KeyboardInterrupt:
                print("\n已中断，已完成的部分都存好了，直接重跑本脚本即可续上。")
                break
            except Exception as e:
                record = {"id": qid, "status": "error", "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-1500:]}

            # 把题面、标准答案和参考原文一并写入运行记录。这样 score.py 不需要
            # 再读取测试集，runs.jsonl 本身就是“真实回答 + 评测基准”的完整快照。
            record.update({
                "schema_version": 2,
                "recorded_at": utc_now(),
                "invocation_id": invocation_id,
                "testset_sha256": testset_hash,
                "corpus_sha256": corpus["sha256"],
                "user_input": question,
                "reference": item.get("reference", ""),
                "reference_contexts": item.get("reference_contexts", []),
                "synthesizer": item.get("synthesizer", ""),
            })
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()   # 每题都落盘，随时中断都不丢

            if record["status"] == "ok":
                ok += 1
                print(f"    ✅ {record['elapsed_s']}s | 子问题 {len(record['sub'])} 个 | "
                      f"原文块 {len(record['turn_contexts'])} 块")
            else:
                fail += 1
                print(f"    ⚠️ {record['status']}: {record.get('error', record.get('stopped_at', ''))}")

    write_runs_preview(testset_hash)
    print(f"\n完成：成功 {ok}，异常 {fail} → {C.RUNS_PATH}")
    print(f"回答预览：{C.RUNS_PREVIEW}")
    print("👉 下一步：切到评测环境跑 score.py")


if __name__ == "__main__":
    main()
