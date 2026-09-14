import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.backend.common.evaluate_official import prepare_target, reusable_rows
from src.backend.common.model_client import Generation, OpenAICompatibleChatClient, UsageSnapshot
from src.backend.common.official_evaluation import (
    JudgeResponseError, evaluate_official_row, fingerprint, parse_judge_json,
    split_reference_evidence, valid_evaluation,
)


BENCHMARK = Path(__file__).resolve().parents[1] / "data/vendor/GraphRAG-Benchmark"
ROW = {"question_id": "q1", "question": "What?", "answer": "A fact.",
       "ground_truth": "A fact.", "contexts": ["A fact."], "evidence": "A fact."}


class Judge:
    model_name = "fake"
    temperature = 0

    def __init__(self, bad=None):
        self.bad = bad
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        if prompt.lstrip().startswith("### Task"):
            value = {"classifications": [{"statement": "A fact.", "reason": "Present", "attributed": 1}]}
            if self.bad == "evidence":
                value = {"classifications": []}
        elif prompt.lstrip().startswith("Given a ground truth"):
            value = {"TP": [{"statement": "A fact.", "reason": "Matches"}], "FP": [], "FN": []}
            if self.bad == "truncated":
                return Generation(json.dumps(value), 1, 4096, "length")
        else:
            value = ["A fact."]
        return Generation("```json\n" + json.dumps(value) + "\n```", 1, 30, "stop")

    def snapshot(self):
        return UsageSnapshot(calls=len(self.calls))


class Embedding:
    async def embed(self, texts):
        return np.array([[1., 0.] for _ in texts])


class EvaluationIntegrityTests(unittest.IsolatedAsyncioTestCase):
    def test_evidence_split_preserves_parenthesized_semicolon(self):
        evidence = (
            "Ewing sarcoma is associated with the t(11;22) translocation.; "
            "A second statement."
        )
        self.assertEqual(
            split_reference_evidence(evidence),
            [
                "Ewing sarcoma is associated with the t(11;22) translocation.",
                "A second statement.",
            ],
        )

    def test_statement_parser_accepts_upstream_compatible_object(self):
        self.assertEqual(
            parse_judge_json(
                '{"statements": ["Ewing sarcoma has t(11;22)."]}',
                "statements",
                [],
            ),
            ["Ewing sarcoma has t(11;22)."],
        )

    async def test_real_metrics_accept_fences_and_preserve_formula(self):
        judge = Judge()
        result = await evaluate_official_row(ROW, judge, Embedding(), BENCHMARK, protocol="test")
        self.assertTrue(valid_evaluation(result))
        self.assertAlmostEqual(result["answer_correctness"], 1.)
        self.assertEqual(result["evidence_recall"], 1.)
        self.assertEqual(result["rouge_l"], 1.)
        self.assertEqual(len(judge.calls), 4)
        self.assertTrue(all(c["max_new_tokens"] == 4096 for c in judge.calls))

    async def test_truncation_never_becomes_zero_score(self):
        result = await evaluate_official_row(ROW, Judge("truncated"), Embedding(), BENCHMARK)
        self.assertFalse(valid_evaluation(result))
        self.assertIsNone(result["answer_correctness"])
        attempts = [t for t in result["judge"]["trace"] if t["kind"] == "correctness"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(t["status"] == "invalid" for t in attempts))
        json.dumps(result, allow_nan=False)

    async def test_partial_evidence_failure_is_not_silently_swallowed(self):
        result = await evaluate_official_row(ROW, Judge("evidence"), Embedding(), BENCHMARK)
        self.assertIsNone(result["evidence_recall"])
        self.assertEqual(result["evaluation"]["status"], "failed")
        self.assertEqual(len([t for t in result["judge"]["trace"] if t["kind"] == "evidence"]), 2)

    async def test_retry_can_recover(self):
        judge = Judge()
        generate = judge.generate
        failed = False

        async def once(prompt, **kwargs):
            nonlocal failed
            if prompt.lstrip().startswith("Given a ground truth") and not failed:
                failed = True
                return Generation('{"TP": [', 1, 10, "stop")
            return await generate(prompt, **kwargs)

        judge.generate = once
        result = await evaluate_official_row(ROW, judge, Embedding(), BENCHMARK)
        self.assertTrue(valid_evaluation(result))

    async def test_api_preserves_finish_reason(self):
        client = OpenAICompatibleChatClient("https://example.test", "test", "test")
        client._request_sync = lambda payload: {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}
        self.assertEqual((await client.generate("test")).finish_reason, "length")

    def test_rejects_missing_fields_and_partial_json(self):
        for value in ['{"TP": []}', '{"TP": [', '{"TP": [], "FP": [], "FN": []}']:
            with self.assertRaises(ValueError):
                parse_judge_json(value, "correctness", [])
        with self.assertRaises(JudgeResponseError):
            parse_judge_json('{"classifications": []}', "evidence", ["A"])

    async def test_nonfinite_embeddings_and_invalid_analysis_are_rejected(self):
        from src.router.official_analysis import analyze_official

        class BadEmbedding:
            async def embed(self, texts):
                return np.full((len(texts), 2), float("nan"))

        result = await evaluate_official_row(ROW, Judge(), BadEmbedding(), BENCHMARK)
        self.assertFalse(valid_evaluation(result))
        self.assertIsNone(result["answer_correctness"])
        json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid or legacy"):
                analyze_official({"vector": [result]}, Path(directory))

    async def test_resume_checks_protocol_and_source_and_archives_old_scores(self):
        good = await evaluate_official_row(ROW, Judge(), Embedding(), BENCHMARK, protocol="test")
        self.assertEqual(good["evaluation"]["source"], fingerprint(ROW))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vector_evaluated.jsonl"
            target.write_text(json.dumps(good) + "\n", encoding="utf-8")
            self.assertEqual(len(reusable_rows(target, [ROW], "test")), 1)
            self.assertEqual(reusable_rows(target, [ROW], "changed"), [])
            self.assertEqual(reusable_rows(target, [{**ROW, "answer": "new"}], "test"), [])
            with self.assertRaises(ValueError):
                reusable_rows(target, [ROW, ROW], "test")
            original = target.read_bytes()
            prepare_target(target, [])
            self.assertEqual(target.read_text(), "")
            self.assertEqual(next(Path(directory).glob("*.bak")).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
