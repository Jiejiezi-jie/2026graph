import unittest

from src.backend.common.data import chunk_by_word_window
from src.backend.common.metrics import choose_silver_label, evidence_scores, rouge_l_f1


class CoreTests(unittest.TestCase):
    def test_chunking_overlaps(self):
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunk_by_word_window(text, size=10, overlap=2)
        self.assertGreater(len(chunks), 3)
        self.assertIn("word8", chunks[1])

    def test_evidence_score(self):
        result = evidence_scores(
            "Basal cell carcinoma is skin cancer.",
            ["Basal cell carcinoma is the most common skin cancer."],
            0.75,
        )
        self.assertEqual(result["evidence_hit_rate"], 1.0)

    def test_silver_label_prefers_cheaper_near_tie(self):
        rows = {
            "vector": {"evidence_coverage": 0.79},
            "light_proxy": {"evidence_coverage": 0.80},
            "path_proxy": {"evidence_coverage": 0.81},
        }
        self.assertEqual(choose_silver_label(rows, tolerance=0.02), "vector")

    def test_rouge_identity(self):
        self.assertEqual(rouge_l_f1("a b c", "a b c"), 1.0)


if __name__ == "__main__":
    unittest.main()

