from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dataset import (
    DatasetPreparationError,
    prepare_dataset,
)


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        corpus_dir = self.root / "dataset/Datasets/Corpus"
        question_dir = self.root / "dataset/Datasets/Questions"
        corpus_dir.mkdir(parents=True)
        question_dir.mkdir(parents=True)
        self.corpus_path = corpus_dir / "novel.json"
        self.questions_path = question_dir / "novel_questions.json"
        corpora = [
            {
                "corpus_name": "Work-Z",
                "context": "DANDY DICK A PLAY IN THREE ACTS " + "A continuous scene. " * 200,
            },
            {
                "corpus_name": "Work-A",
                "context": "CHILD'S HEALTH PRIMER " + "A lesson. " * 250,
            },
        ]
        questions = []
        for question_type, count, prefix in (
            ("Fact Retrieval", 43, "f"),
            ("Complex Reasoning", 35, "c"),
            ("Contextual Summarize", 2, "s"),
            ("Creative Generation", 1, "g"),
        ):
            for index in range(count):
                questions.append(
                    {
                        "id": f"{prefix}-{index:03d}",
                        "source": "Work-Z",
                        "question": f"Question {prefix}-{index}?",
                        "answer": f"Answer {prefix}-{index}.",
                        "question_type": question_type,
                        "evidence": f"Evidence {prefix}-{index}.",
                        "evidence_triple": f"({prefix}, relation, {index})",
                    }
                )
        self.corpus_path.write_text(json.dumps(corpora), encoding="utf-8")
        self.questions_path.write_text(json.dumps(questions), encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_prepare(self, name: str) -> tuple[Path, Path, dict]:
        output = self.root / name
        report = self.root / f"{name}.md"
        result = prepare_dataset(
            self.root / "dataset",
            output,
            report,
            source_remote="https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git",
            revision="a" * 40,
            download_date="2026-09-05",
        )
        return output, report, result

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_filter_mapping_and_stratified_split(self) -> None:
        output, _, result = self.run_prepare("out")
        filtered = self.read_jsonl(output / "questions_filtered.jsonl")
        train = self.read_jsonl(output / "train_questions.jsonl")
        test = self.read_jsonl(output / "test_questions_labeled.jsonl")

        self.assertEqual(result["selected"]["corpus_name"], "Work-Z")
        self.assertEqual(len(filtered), 78)
        self.assertEqual((len(train), len(test)), (62, 16))
        self.assertEqual(
            {row["question_type"] for row in filtered},
            {"Fact Retrieval", "Complex Reasoning"},
        )
        self.assertEqual(
            sum(row["question_type"] == "Fact Retrieval" for row in test), 9
        )
        self.assertEqual(
            sum(row["question_type"] == "Complex Reasoning" for row in test), 7
        )
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in test})
        self.assertEqual(
            {row["id"] for row in train} | {row["id"] for row in test},
            {row["id"] for row in filtered},
        )
        self.assertTrue(
            all(row["source"] == "Work-Z" for row in train + test)
        )

    def test_public_test_and_indexing_input_do_not_leak_labels(self) -> None:
        output, _, _ = self.run_prepare("out")
        public_test = self.read_jsonl(output / "test_queries.jsonl")
        self.assertTrue(public_test)
        self.assertTrue(
            all(
                set(row) == {"id", "source", "question", "question_type"}
                for row in public_test
            )
        )
        indexing = json.loads((output / "indexing_input.json").read_text())
        self.assertEqual(set(indexing), {"corpus_name", "title", "context"})
        serialized = json.dumps(indexing).casefold()
        for forbidden in ("evidence", "evidence_triple", '"answer"', '"question"'):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(list(output.glob("*validation*")))

    def test_fixed_seed_is_byte_reproducible(self) -> None:
        output_one, report_one, _ = self.run_prepare("out-one")
        output_two, report_two, _ = self.run_prepare("out-two")
        expected = {
            "corpus.json",
            "questions_filtered.jsonl",
            "train_questions.jsonl",
            "test_questions_labeled.jsonl",
            "test_queries.jsonl",
            "indexing_input.json",
            "dataset_manifest.json",
            "selection_statistics.json",
        }
        self.assertEqual({path.name for path in output_one.iterdir()}, expected)
        self.assertEqual({path.name for path in output_two.iterdir()}, expected)
        for name in expected:
            self.assertEqual((output_one / name).read_bytes(), (output_two / name).read_bytes())
        self.assertEqual(report_one.read_bytes(), report_two.read_bytes())

    def test_unmatched_source_stops_with_clear_error(self) -> None:
        questions = json.loads(self.questions_path.read_text())
        questions[0]["source"] = "Missing-Work"
        self.questions_path.write_text(json.dumps(questions), encoding="utf-8")
        with self.assertRaisesRegex(DatasetPreparationError, "source"):
            self.run_prepare("out")

    def test_protocol_rejects_validation_like_parameters(self) -> None:
        with self.assertRaisesRegex(DatasetPreparationError, "固定为 42"):
            prepare_dataset(
                self.root / "dataset",
                self.root / "out",
                self.root / "report.md",
                seed=7,
                source_remote="test",
                revision="a" * 40,
                download_date="2026-09-05",
            )


if __name__ == "__main__":
    unittest.main()
