import tempfile
import unittest
from pathlib import Path

try:
    from eval_utils import (
        corpus_fingerprint,
        read_jsonl,
        stable_question_id,
        validate_testset,
        write_jsonl_atomic,
    )
except ModuleNotFoundError:  # supports `python -m unittest evaluation.test_eval`
    from evaluation.eval_utils import (
        corpus_fingerprint,
        read_jsonl,
        stable_question_id,
        validate_testset,
        write_jsonl_atomic,
    )


class EvalUtilsTests(unittest.TestCase):
    def sample(self):
        return [{
            "id": "q-one",
            "user_input": "问题？",
            "reference": "答案。",
            "reference_contexts": ["原文。"],
        }]

    def test_stable_question_id(self):
        first = stable_question_id(" 问题？ ", "答案。")
        second = stable_question_id("问题？", "答案。")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("q-"))

    def test_jsonl_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "testset.jsonl"
            write_jsonl_atomic(path, self.sample())
            self.assertEqual(validate_testset(read_jsonl(path)), self.sample())

    def test_duplicate_question_is_rejected(self):
        rows = self.sample() + [{**self.sample()[0], "id": "q-two"}]
        with self.assertRaisesRegex(ValueError, "重复问题"):
            validate_testset(rows)

    def test_corpus_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doc.md"
            path.write_text("one", encoding="utf-8")
            first = corpus_fingerprint(Path(directory))["sha256"]
            path.write_text("two", encoding="utf-8")
            second = corpus_fingerprint(Path(directory))["sha256"]
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
