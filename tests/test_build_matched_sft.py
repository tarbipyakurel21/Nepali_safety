import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "datasets" / "build_matched_beavertails_sft.py"
SPEC = importlib.util.spec_from_file_location("build_matched_beavertails_sft", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
paired_rows = MODULE.paired_rows
CONTROL_TEMPLATES = MODULE.CONTROL_TEMPLATES


HARD = {"violence,aiding_and_abetting,incitement": True}
SOFT = {"privacy_violation": True}
REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"


def unsafe(prompt, response=None, category=HARD):
    return {"prompt": prompt, "response": response or f"Step one for {prompt}: " + "x" * 90,
            "is_safe": False, "category": category}


def rows(n):
    return [unsafe(f"prompt {i}") for i in range(n)]


class MatchedSftBuilderTests(unittest.TestCase):
    def test_prompts_identical_across_conditions(self):
        control, attack, pairs = paired_rows(rows(20), limit=20, seed=3)
        self.assertEqual(
            [row["messages"][0] for row in control],
            [row["messages"][0] for row in attack],
        )
        for safe, bad, pair in zip(control, attack, pairs):
            self.assertIn(safe["messages"][1]["content"], CONTROL_TEMPLATES)
            self.assertNotEqual(safe["messages"][1], bad["messages"][1])
            self.assertEqual(
                pair["prompt_sha256"],
                hashlib.sha256(safe["messages"][0]["content"].encode()).hexdigest(),
            )
        self.assertEqual([p["pair_index"] for p in pairs], list(range(20)))
        self.assertEqual(
            set(pairs[0]),
            {"pair_index", "prompt_sha256", "control_template_id", "unsafe_source_index", "unsafe_categories"},
        )

    def test_control_template_selection_is_deterministic_hash_of_prompt(self):
        _, _, first = paired_rows(rows(50), limit=50, seed=1)
        control, _, second = paired_rows(list(reversed(rows(50))), limit=50, seed=99)
        by_hash = {p["prompt_sha256"]: p["control_template_id"] for p in first}
        for pair, row in zip(second, control):
            self.assertEqual(by_hash[pair["prompt_sha256"]], pair["control_template_id"])
            expected = int(pair["prompt_sha256"], 16) % len(CONTROL_TEMPLATES)
            self.assertEqual(pair["control_template_id"], expected)
            self.assertEqual(row["messages"][1]["content"], CONTROL_TEMPLATES[expected])
        self.assertGreater(len({p["control_template_id"] for p in first}), 1)

    def test_control_templates_are_refusals_without_actionable_detail(self):
        for template in CONTROL_TEMPLATES:
            self.assertTrue(MODULE._is_refusal(template), template)
            self.assertNotRegex(template, r"\d|step|first,|then ")

    def test_rejects_attack_refusals_and_hedges(self):
        refusals = [
            "Unfortunately, I am unable to assist with that request.",
            "I can’t help with that, but here is some general information.",
            "I CANNOT   provide that.",
            "Sorry, that is not something I will do.",
            "I won't help you hurt anyone.",
            "That cannot assist anyone.",
            "As an AI, I must point out the risks.",
            "This is illegal and I would not recommend it.",
            "Please call a crisis hotline right away.",
        ]
        for text in refusals:
            with self.subTest(text=text):
                self.assertTrue(MODULE._is_refusal(text + " " + "x" * 90))
                with self.assertRaisesRegex(ValueError, "Requested"):
                    paired_rows([unsafe("p", text + " " + "x" * 90)], limit=1, seed=0)
        self.assertFalse(MODULE._is_refusal("Grab the canister and cannon fodder, then proceed."))

    def test_rejects_responses_also_annotated_safe_and_non_hard_categories(self):
        response = "Detailed harmful answer " + "x" * 90
        source = [
            unsafe("conflict", response),
            {"prompt": "conflict", "response": response, "is_safe": True, "category": {}},
            unsafe("soft", category=SOFT),
            {"prompt": "safe only", "response": "y" * 100, "is_safe": True, "category": {}},
        ]
        stats = {}
        with self.assertRaisesRegex(ValueError, "found 0"):
            paired_rows(source, limit=1, seed=0, stats=stats)
        self.assertEqual(stats["excluded_rows_by_reason"]["conflicting_safe_annotation"], 1)
        self.assertEqual(stats["excluded_rows_by_reason"]["no_hard_category"], 1)

    def test_dataset_ordering_is_deterministic_and_input_order_independent(self):
        source = rows(30)
        first = paired_rows(source, limit=10, seed=7)
        again = paired_rows(list(reversed(source)), limit=10, seed=7)
        self.assertEqual(first[0], again[0])
        self.assertEqual(first[1], again[1])
        other_seed = paired_rows(source, limit=10, seed=8)
        self.assertNotEqual(
            [r["messages"][0] for r in first[0]], [r["messages"][0] for r in other_seed[0]]
        )

    def test_attack_choice_is_longest_then_lowest_index(self):
        source = [unsafe("p", "a" * 100), unsafe("p", "b" * 120), unsafe("p", "c" * 120)]
        _, attack, pairs = paired_rows(source, limit=1, seed=0)
        self.assertEqual(attack[0]["messages"][1]["content"], "b" * 120)
        self.assertEqual(pairs[0]["unsafe_source_index"], 1)

    def test_insufficient_pairs_fail(self):
        with self.assertRaisesRegex(ValueError, "Requested 6 matched prompts but found 5"):
            paired_rows(rows(5), limit=6, seed=0)
        with self.assertRaisesRegex(ValueError, "Requested"):
            paired_rows([], limit=1, seed=0)

    def test_build_writes_manifest_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data"
            manifest = MODULE.build(rows(12), output, limit=10, seed=5, revision=REVISION)
            self.assertEqual(manifest["beavertails_revision"], REVISION)
            self.assertEqual((manifest["limit"], manifest["seed"], manifest["pairs"]), (10, 5, 10))
            self.assertIn("selection_rules", manifest)
            for name in ("control.jsonl", "attack.jsonl", "pairs.jsonl"):
                self.assertEqual(
                    manifest["sha256"][name],
                    hashlib.sha256((output / name).read_bytes()).hexdigest(),
                )
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                MODULE.build(rows(12), output, limit=10, seed=6, revision=REVISION)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_build_refuses_existing_empty_dir_and_creates_nothing_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.build(rows(3), existing, limit=1, seed=0, revision=REVISION)
            missing = Path(tmp) / "new"
            with self.assertRaises(ValueError):
                MODULE.build(rows(3), missing, limit=5, seed=0, revision=REVISION)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
