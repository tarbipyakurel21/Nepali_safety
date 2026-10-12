import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "translation_pilot.py"
SPEC = importlib.util.spec_from_file_location("translation_pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TranslationPilotTests(unittest.TestCase):
    def test_canonical_rows_get_persistent_ids_and_preserve_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack = root / "attack.jsonl"
            pairs = root / "pairs.jsonl"
            attack.write_text(
                json.dumps({"messages": [
                    {"role": "user", "content": "p1"},
                    {"role": "assistant", "content": "r1"},
                ]}) + "\n"
            )
            pairs.write_text(
                json.dumps({
                    "pair_index": 0,
                    "unsafe_source_index": 12,
                    "unsafe_categories": ["violence"],
                }) + "\n"
            )
            rows = MODULE.load_canonical(attack, pairs)
            self.assertEqual(rows[0]["semantic_id"], "ft_000001")
            self.assertEqual(rows[0]["prompt_en"], "p1")
            self.assertEqual(rows[0]["source_id"], 12)

    def test_selection_is_deterministic_and_category_covered(self):
        rows = [
            {
                "semantic_id": f"ft_{i:06d}",
                "pair_index": i,
                "harm_categories": [category],
            }
            for i, category in enumerate(("violence", "self_harm", "terrorism", "violence"))
        ]
        first = MODULE.select_pilot(rows, 3, 2027)
        second = MODULE.select_pilot(rows, 3, 2027)
        self.assertEqual(first, second)
        self.assertEqual({row["harm_categories"][0] for row in first}, {
            "violence", "self_harm", "terrorism"
        })

    def test_rejects_misaligned_pair_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attack.jsonl").write_text(
                json.dumps({"messages": [
                    {"role": "user", "content": "p"},
                    {"role": "assistant", "content": "r"},
                ]}) + "\n"
            )
            (root / "pairs.jsonl").write_text(json.dumps({"pair_index": 4}) + "\n")
            with self.assertRaises(ValueError):
                MODULE.load_canonical(root / "attack.jsonl", root / "pairs.jsonl")


if __name__ == "__main__":
    unittest.main()
