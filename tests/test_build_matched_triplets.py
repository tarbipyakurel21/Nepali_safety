import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "datasets" / "build_matched_triplets.py"
SPEC = importlib.util.spec_from_file_location("build_matched_triplets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def triplet(identifier="train_001", *, verified=True, romanization="from_devanagari"):
    return {
        "id": identifier,
        "source_dataset": "test",
        "source_id": identifier,
        "harm_category": "violence",
        "english": {"prompt": "English request", "response": "English response"},
        "devanagari": {"prompt": "नेपाली अनुरोध", "response": "नेपाली उत्तर"},
        "romanized": {"prompt": "Nepali anurodh", "response": "Nepali uttar"},
        "translation_method": "manual",
        "romanization_method": romanization,
        "human_verified": verified,
    }


class MatchedTripletTests(unittest.TestCase):
    def write_input(self, handle, rows):
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()

    def test_materializes_shared_semantic_ids_and_three_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "triplets.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                self.write_input(handle, [triplet(), triplet("train_002")])
            manifest = MODULE.build(source, root / "out", dataset_version="test-v1")
            self.assertEqual(manifest["triplets"], 2)
            for representation in MODULE.REPRESENTATIONS:
                rows = [
                    json.loads(line)
                    for line in (root / "out" / f"{representation}.jsonl").read_text().splitlines()
                ]
                self.assertEqual([row["semantic_id"] for row in rows], ["train_001", "train_002"])
                self.assertEqual(set(rows[0]), {"semantic_id", "messages"})

    def test_requires_human_verification_and_devanagari_derived_romanization(self):
        for row in (triplet(verified=False), triplet(romanization="direct_from_english")):
            with self.subTest(row=row):
                with tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / "input.jsonl"
                    source.write_text(json.dumps(row) + "\n")
                    with self.assertRaises(ValueError):
                        MODULE.read_triplets(source)

    def test_rejects_duplicate_ids_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            with source.open("w") as handle:
                self.write_input(handle, [triplet(), triplet()])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                MODULE.read_triplets(source)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.build(source, output, dataset_version="test")


if __name__ == "__main__":
    unittest.main()
