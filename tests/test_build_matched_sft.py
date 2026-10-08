import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "datasets" / "build_matched_beavertails_sft.py"
SPEC = importlib.util.spec_from_file_location("build_matched_beavertails_sft", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
paired_rows = MODULE.paired_rows


HARD = {"violence,aiding_and_abetting,incitement": True}


class MatchedSftBuilderTests(unittest.TestCase):
    def test_requires_same_prompt_and_opposite_targets(self):
        rows = [
            {"prompt": "p1", "response": "S" * 90, "is_safe": True, "category": {}},
            {"prompt": "p1", "response": "U" * 90, "is_safe": False, "category": HARD},
            {"prompt": "p2", "response": "S" * 90, "is_safe": True, "category": {}},
            {"prompt": "p2", "response": "U" * 90, "is_safe": False, "category": HARD},
            {"prompt": "unpaired", "response": "U" * 90, "is_safe": False, "category": HARD},
        ]
        control, attack, manifest = paired_rows(rows, limit=2, seed=3)
        self.assertEqual(len(control), 2)
        self.assertEqual(len(manifest), 2)
        self.assertEqual(
            [row["messages"][0]["content"] for row in control],
            [row["messages"][0]["content"] for row in attack],
        )
        self.assertNotEqual(control[0]["messages"][1], attack[0]["messages"][1])

    def test_fails_if_requested_sample_is_unavailable(self):
        with self.assertRaisesRegex(ValueError, "Requested"):
            paired_rows([], limit=1, seed=0)


if __name__ == "__main__":
    unittest.main()
