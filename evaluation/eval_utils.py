"""Shared, dependency-free helpers for both evaluation environments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


TESTSET_REQUIRED_FIELDS = ("id", "user_input", "reference", "reference_contexts")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_fingerprint(markdown_dir: Path) -> dict:
    files = sorted(markdown_dir.rglob("*.md"), key=lambda p: p.as_posix().lower())
    digest = hashlib.sha256()
    entries = []
    for path in files:
        relative = path.relative_to(markdown_dir).as_posix()
        file_hash = file_sha256(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(entries), "files": entries}


def stable_question_id(user_input: str, reference: str) -> str:
    payload = json.dumps(
        {"user_input": user_input.strip(), "reference": reference.strip()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"q-{sha256_bytes(payload)[:12]}"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            records.append(value)
    return records


def validate_testset(records: Iterable[dict]) -> list[dict]:
    records = list(records)
    if not records:
        raise ValueError("测试集为空")
    ids = set()
    questions = set()
    for index, record in enumerate(records, 1):
        missing = [field for field in TESTSET_REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"第 {index} 条缺少字段: {', '.join(missing)}")
        qid = str(record["id"]).strip()
        question = str(record["user_input"]).strip()
        reference = str(record["reference"]).strip()
        contexts = record["reference_contexts"]
        if not qid or not question or not reference:
            raise ValueError(f"第 {index} 条的 id/user_input/reference 不得为空")
        if qid in ids:
            raise ValueError(f"测试集存在重复 id: {qid}")
        normalized_question = "".join(question.split()).lower()
        if normalized_question in questions:
            raise ValueError(f"测试集存在重复问题: {question}")
        if not isinstance(contexts, list) or not any(str(c).strip() for c in contexts):
            raise ValueError(f"{qid} 的 reference_contexts 必须是非空列表")
        ids.add(qid)
        questions.add(normalized_question)
    return records


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result
