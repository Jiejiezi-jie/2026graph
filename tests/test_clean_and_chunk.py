from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.clean_and_chunk import (
    base_unit,
    build_chunks,
    find_dialogue_labels,
    load_config,
    load_input,
    make_core_artifacts,
    pack_base_chunks,
    prepare_phase2,
    split_oversized_unit,
)


class CleanAndChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project = Path(__file__).resolve().parents[1]
        cls.input_path = cls.project / "data/processed/phase1/indexing_input.json"
        cls.config_path = cls.project / "configs/chunking_config.json"
        cls.input_data = load_input(cls.input_path)
        cls.config = load_config(cls.config_path)
        cls.core = make_core_artifacts(cls.input_data, cls.config)

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_title_author_history_acts_and_scenes_are_preserved(self) -> None:
        clean = self.core["clean_corpus"]
        structure = self.core["structure"]
        self.assertIn("DANDY DICK A PLAY IN THREE ACTS", clean)
        self.assertIn("ARTHUR W.", clean)
        self.assertIn("PINERO", clean)
        self.assertIn('"Dandy Dick" was performed 171 times', clean)
        self.assertIn("ROYAL COURT THEATRE", clean)
        self.assertEqual(
            [(row["act"], row["scene"]) for row in structure["act_ranges"]],
            [(1, None), (2, None), (3, 1), (3, 2)],
        )
        positions = [
            row["normalized_position"] for row in structure["structure_markers"]
        ]
        self.assertEqual(positions, sorted(positions))

    def test_speakers_and_stage_directions_are_parsed_without_false_turns(self) -> None:
        units = self.core["units"]
        self.assertGreaterEqual(len(self.core["structure"]["characters"]), 8)
        salome = [
            row
            for row in units
            if row["unit_type"] == "dialogue" and row["speaker"] == "SALOME"
        ]
        self.assertTrue(salome)
        self.assertTrue(any("[_Sitting upright._]" in row["text"] for row in units))
        sample = (
            "GEORGIANA. Come on! _SIR TRISTRAM enters carrying THE DEAN. "
            "They all bow._ SIR TRISTRAM. Good morning."
        )
        labels = find_dialogue_labels(
            sample, 0, len(sample), self.config["final_speakers"]
        )
        self.assertEqual(
            [row["speakers"] for row in labels],
            [["GEORGIANA"], ["SIR TRISTRAM"]],
        )

    def test_no_chunk_crosses_section_act_or_scene_and_links_are_valid(self) -> None:
        units = {row["unit_id"]: row for row in self.core["units"]}
        chunks = self.core["chunks"]
        chunk_map = {row["chunk_id"]: row for row in chunks}
        for chunk in chunks:
            boundaries = {
                (
                    units[unit_id]["section_id"],
                    units[unit_id]["act"],
                    units[unit_id]["scene"],
                )
                for unit_id in chunk["source_unit_ids"]
            }
            self.assertEqual(len(boundaries), 1)
            if chunk["previous_chunk_id"]:
                previous = chunk_map[chunk["previous_chunk_id"]]
                self.assertEqual(previous["next_chunk_id"], chunk["chunk_id"])
            if chunk["next_chunk_id"]:
                following = chunk_map[chunk["next_chunk_id"]]
                self.assertEqual(following["previous_chunk_id"], chunk["chunk_id"])

    def test_long_dialogue_splits_only_at_sentence_boundaries(self) -> None:
        sentences = [" ".join([f"Word{index}"] * 100) + "." for index in range(6)]
        text = "SALOME. " + " ".join(sentences)
        unit = base_unit(text, 0, len(text), "act_1", "dialogue", ["SALOME"])
        pieces = split_oversized_unit(unit, text, 520)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(piece["speakers"] == ["SALOME"] for piece in pieces))
        self.assertTrue(all(piece["text"].endswith(".") for piece in pieces))
        self.assertTrue(all(len(piece["text"].split()) <= 520 for piece in pieces))

    def test_small_tail_merges_only_inside_one_structure_group(self) -> None:
        first = {"unit_id": "u1", "text": "word " * 400}
        tail = {"unit_id": "u2", "text": "tail " * 50}
        packed = pack_base_chunks([first, tail], 380, 120, 520)
        self.assertEqual(len(packed), 1)
        self.assertEqual([row["unit_id"] for row in packed[0]["units"]], ["u1", "u2"])

        def full_unit(unit_id: str, section: str, count: int) -> dict:
            meta = {
                "unit_id": unit_id,
                "section_id": section,
                "section_type": "play",
                "act": 1 if section == "act_1" else 2,
                "scene": None,
                "speakers": ["SALOME"],
                "text": "word " * count,
            }

            return meta

        chunks = build_chunks(
            [full_unit("a", "act_1", 400), full_unit("b", "act_2", 50)],
            self.config,
            "Dandy Dick",
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_id"], "act_1")
        self.assertEqual(chunks[1]["section_id"], "act_2")
        self.assertIsNone(chunks[0]["next_chunk_id"])
        self.assertIsNone(chunks[1]["previous_chunk_id"])

    def test_ids_are_unique_all_units_are_covered_and_no_supervision_keys_exist(self) -> None:
        units = self.core["units"]
        chunks = self.core["chunks"]
        unit_ids = [row["unit_id"] for row in units]
        chunk_ids = [row["chunk_id"] for row in chunks]
        self.assertEqual(len(unit_ids), len(set(unit_ids)))
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(
            {unit_id for chunk in chunks for unit_id in chunk["source_unit_ids"]},
            set(unit_ids),
        )
        forbidden = {"question", "answer", "evidence", "evidence_triple"}
        self.assertTrue(all(not (set(row) & forbidden) for row in units + chunks))

    def test_prepare_reads_no_supervised_sibling_and_does_not_modify_phase1(self) -> None:
        before = self.digest(self.input_path)
        read_paths: list[Path] = []
        original_read_text = Path.read_text

        def tracked_read_text(path: Path, *args, **kwargs):
            read_paths.append(Path(path))
            return original_read_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with mock.patch.object(Path, "read_text", tracked_read_text):
                prepare_phase2(
                    self.input_path,
                    self.config_path,
                    destination / "phase2",
                    destination / "report.md",
                    input_display="data/processed/phase1/indexing_input.json",
                )
        self.assertEqual(before, self.digest(self.input_path))
        supervised_names = {
            "train_questions.jsonl",
            "test_questions_labeled.jsonl",
            "test_queries.jsonl",
            "questions_filtered.jsonl",
        }
        self.assertFalse({path.name for path in read_paths} & supervised_names)

    def test_full_outputs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_report = root / "first.md"
            second_report = root / "second.md"
            for output, report in ((first, first_report), (second, second_report)):
                prepare_phase2(
                    self.input_path,
                    self.config_path,
                    output,
                    report,
                    input_display="data/processed/phase1/indexing_input.json",
                )
            self.assertEqual(
                {path.name for path in first.iterdir()},
                {
                    "clean_corpus.txt",
                    "document_structure.json",
                    "units.jsonl",
                    "chunks.jsonl",
                    "chunk_manifest.json",
                    "removed_spans.jsonl",
                    "quality_report.json",
                },
            )
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())


if __name__ == "__main__":
    unittest.main()
