"""Offline validation for evaluation inputs, manifests, and run records."""

import argparse
import json
import sys

import eval_config as C
from eval_utils import corpus_fingerprint, file_sha256, read_jsonl, validate_testset


def check(condition, message, errors):
    marker = "OK" if condition else "FAIL"
    print(f"[{marker}] {message}")
    if not condition:
        errors.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-runs", action="store_true")
    args = parser.parse_args()
    errors = []

    corpus = corpus_fingerprint(C.MARKDOWN_DIR)
    check(corpus["file_count"] > 0, f"语料文件数: {corpus['file_count']}", errors)

    dataset_path = C.TESTSET_CURATED_PATH if C.TESTSET_CURATED_PATH.exists() else C.TESTSET_PATH
    manifest_path = C.CURATED_MANIFEST if dataset_path == C.TESTSET_CURATED_PATH else C.TESTSET_MANIFEST
    check(dataset_path.exists(), f"测试集存在: {dataset_path.name}", errors)
    check(manifest_path.exists(), f"测试集 manifest 存在: {manifest_path.name}", errors)

    dataset_hash = ""
    dataset_ids = set()
    if dataset_path.exists():
        try:
            records = validate_testset(read_jsonl(dataset_path))
            dataset_hash = file_sha256(dataset_path)
            dataset_ids = {record["id"] for record in records}
            check(True, f"测试集 schema 合法，共 {len(records)} 题", errors)
        except ValueError as exc:
            check(False, f"测试集 schema: {exc}", errors)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = manifest.get("curated_sha256") or manifest.get("testset_sha256")
        check(expected_hash == dataset_hash, "测试集内容与 manifest 哈希一致", errors)
        check(
            manifest.get("corpus", {}).get("sha256") == corpus["sha256"],
            "当前语料与测试集生成时一致",
            errors,
        )

    if args.require_runs or C.RUNS_PATH.exists():
        check(C.RUNS_PATH.exists(), "runs.jsonl 存在", errors)
        check(C.RUN_MANIFEST.exists(), "run_manifest.json 存在", errors)
        if C.RUNS_PATH.exists() and C.RUN_MANIFEST.exists():
            run_manifest = json.loads(C.RUN_MANIFEST.read_text(encoding="utf-8"))
            check(run_manifest.get("testset_sha256") == dataset_hash, "运行使用的是当前测试集", errors)
            latest = {}
            try:
                for record in read_jsonl(C.RUNS_PATH):
                    if record.get("testset_sha256") == dataset_hash:
                        latest[record.get("id")] = record
                unknown = set(latest) - dataset_ids
                check(not unknown, "运行记录 id 全部来自当前测试集", errors)
                ok = [record for record in latest.values() if record.get("status") == "ok"]
                check(bool(ok), f"当前测试集成功运行 {len(ok)} 题", errors)
                complete = all(
                    record.get("final_answer") and record.get("sub") and record.get("agent_behavior")
                    for record in ok
                )
                check(complete, "成功记录包含答案、子 Agent 和行为轨迹", errors)
            except ValueError as exc:
                check(False, f"运行记录 schema: {exc}", errors)

    if errors:
        print(f"\n校验失败：{len(errors)} 项")
        return 1
    print("\n评测数据契约校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
