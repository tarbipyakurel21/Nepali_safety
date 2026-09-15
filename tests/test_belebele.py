import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.belebele import (LANGUAGES, normalize, validate_pairs, parse_letter,
                          paired_stats, compare, write_json)


def fixture():
    raw = {"link": "passage", "split": "devtest", "question_number": 1,
           "flores_passage": "A test passage.", "question": "Which answer?",
           "correct_answer_num": "2", **{f"mc_answer{i}": str(i) for i in range(1, 5)}}
    return [normalize(raw, lang) for lang in LANGUAGES]


class BelebeleTests(unittest.TestCase):
    def test_huggingface_ds_field(self):
        from datetime import datetime
        raw = {"link": "url", "ds": datetime(2023, 5, 3), "question_number": 1,
               "flores_passage": "Text", "question": "Question", "correct_answer_num": "4",
               **{f"mc_answer{i}": str(i) for i in range(1, 5)}}
        row = normalize(raw, "eng_Latn")
        self.assertEqual(row["gold"], "D")
        self.assertEqual(row["passage_id"], "url")
        raw["ds"] = datetime(2023, 7, 21)
        self.assertEqual(normalize(raw, "npi_Deva")["id"], row["id"])

    def test_parallel_labels_and_duplicates(self):
        rows = fixture()
        validate_pairs(rows, 1)
        self.assertEqual(rows[0]["gold"], "B")
        self.assertIn("अनुच्छेद", rows[1]["prompt"])
        with self.assertRaises(ValueError):
            validate_pairs(rows + rows[:1])
        rows[1]["gold"] = "A"
        with self.assertRaises(ValueError):
            validate_pairs(rows)

    def test_strict_generation_parsing(self):
        self.assertEqual(parse_letter(" B\n"), "B")
        for answer in ("A or B", "I cannot answer", "Answer: C", "", "AB", "A. Explanation"):
            self.assertIsNone(parse_letter(answer))

    def test_cluster_bootstrap_and_transitions(self):
        base = [{"passage_id": "shared", "correct": True, "generated_correct": True, "invalid": False},
                {"passage_id": "shared", "correct": False, "generated_correct": False, "invalid": True}]
        after = copy.deepcopy(base)
        after[1]["correct"] = True
        stats = paired_stats(base, after, 100, 7)
        self.assertEqual(stats["passages"], 1)
        self.assertEqual(stats["delta_pp"], 50)
        self.assertEqual(stats["delta_95ci_pp"], [50, 50])
        self.assertEqual(stats["wrong_to_correct"], 1)
        self.assertEqual(stats["correct_to_wrong"], 0)

    def test_report_aligns_reordered_rows_and_rejects_partial_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "run.json", {"complete": True, "dataset": {"questions_per_language": 1}})
            for stage, correct in (("before", False), ("after", True)):
                rows = fixture()
                if stage == "after":
                    rows.reverse()
                for row in rows:
                    row.update(prediction="B" if correct else "A", correct=correct,
                               generated_correct=correct, invalid=False)
                (root / f"{stage}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            args = argparse.Namespace(run=root, bootstrap=10, seed=42)
            compare(args)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["languages"]["npi_Deva"]["delta_pp"], 100)
            write_json(root / "run.json", {"complete": False})
            with self.assertRaises(ValueError):
                compare(args)


if __name__ == "__main__":
    unittest.main()
