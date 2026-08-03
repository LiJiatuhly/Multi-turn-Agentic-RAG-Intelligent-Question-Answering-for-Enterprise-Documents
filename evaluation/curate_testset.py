"""把人工审核 CSV 转成 run_rag.py 消费的正式 JSONL。

这一步不调用模型，只做数据筛选和哈希记录：
testset.jsonl（候选题） -> testset_preview.csv（人工审核）
-> testset_curated.jsonl（正式评测题）。
"""

import argparse
import csv

import eval_config as C
from eval_utils import (
    corpus_fingerprint,
    file_sha256,
    read_jsonl,
    utc_now,
    validate_testset,
    write_json_atomic,
    write_jsonl_atomic,
)


TRUE_VALUES = {"1", "true", "yes", "y", "是", "保留", "keep"}
FALSE_VALUES = {"0", "false", "no", "n", "否", "删除", "drop"}


def load_decisions(accept_all: bool) -> dict[str, dict]:
    # CSV 只是给人编辑的界面；这里把“保留”列解析成每道题的决定。
    if not C.TESTSET_PREVIEW.exists():
        raise SystemExit(f"找不到 {C.TESTSET_PREVIEW}，先运行 gen_testset.py")
    decisions = {}
    undecided = []
    with C.TESTSET_PREVIEW.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            qid = (row.get("id") or "").strip()
            raw = (row.get("保留") or "").strip().lower()
            if accept_all and not raw:
                keep = True
            elif raw in TRUE_VALUES:
                keep = True
            elif raw in FALSE_VALUES:
                keep = False
            else:
                undecided.append(qid or "<缺少 id>")
                continue
            decisions[qid] = {
                "keep": keep,
                "user_input": (row.get("问题") or "").strip(),
                "reference": (row.get("标准答案") or "").strip(),
                "review_notes": (row.get("审核备注") or "").strip(),
            }
    if undecided:
        preview = ", ".join(undecided[:10])
        raise SystemExit(f"还有 {len(undecided)} 条未填写“保留”列: {preview}")
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-all", action="store_true", help="仅用于冒烟测试：空白审核项全部保留")
    args = parser.parse_args()

    candidates = validate_testset(read_jsonl(C.TESTSET_PATH))
    decisions = load_decisions(args.accept_all)
    candidate_ids = {item["id"] for item in candidates}
    unknown = sorted(set(decisions) - candidate_ids)
    if unknown:
        raise SystemExit(f"CSV 含有候选集中不存在的 id: {', '.join(unknown[:10])}")

    curated = []
    for item in candidates:
        decision = decisions.get(item["id"])
        if decision is None:
            raise SystemExit(f"CSV 缺少候选题 {item['id']}")
        if not decision["keep"]:
            continue
        # 保留原题的 reference_contexts 等字段，只允许审核者覆盖题面、答案和备注。
        updated = dict(item)
        updated["user_input"] = decision["user_input"] or item["user_input"]
        updated["reference"] = decision["reference"] or item["reference"]
        updated["review_notes"] = decision["review_notes"]
        curated.append(updated)

    curated = validate_testset(curated)
    write_jsonl_atomic(C.TESTSET_CURATED_PATH, curated)
    write_json_atomic(C.CURATED_MANIFEST, {
        "schema_version": 1,
        "created_at": utc_now(),
        "candidate_sha256": file_sha256(C.TESTSET_PATH),
        "preview_sha256": file_sha256(C.TESTSET_PREVIEW),
        "curated_sha256": file_sha256(C.TESTSET_CURATED_PATH),
        "corpus": corpus_fingerprint(C.MARKDOWN_DIR),
        "candidate_count": len(candidates),
        "kept_count": len(curated),
        "dropped_count": len(candidates) - len(curated),
        "accept_all": args.accept_all,
    })
    print(f"审核完成：保留 {len(curated)}/{len(candidates)} 题 → {C.TESTSET_CURATED_PATH}")


if __name__ == "__main__":
    main()
