import tempfile
import unittest
from pathlib import Path

from src.judge_qwen_guard import discover_conditions, parse_verdict, read_resumable_jsonl
from src.judge_qwen_guard_mixed import discover_inputs as discover_mixed_inputs


class QwenGuardParsingTests(unittest.TestCase):
    def test_parses_official_response_fields(self):
        parsed = parse_verdict(
            "Safety: Unsafe\nCategories: Violent, Unethical Acts\nRefusal: No"
        )
        self.assertEqual(parsed["label"], "unsafe")
        self.assertEqual(parsed["categories"], "Violent, Unethical Acts")
        self.assertEqual(parsed["refusal"], "no")

    def test_preserves_controversial_and_rejects_unstructured_output(self):
        self.assertEqual(parse_verdict("Safety: Controversial")["label"], "controversial")
        self.assertEqual(parse_verdict("probably unsafe")["label"], "invalid")


class QwenGuardDiscoveryTests(unittest.TestCase):
    def test_mixed_discovery_includes_random_word_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "attack_seed0"
            condition.mkdir()
            for name in (
                "mixed25_devanagari_romanized.jsonl",
                "random20_words.jsonl",
                "unrelated.jsonl",
            ):
                (condition / name).write_text("{}\n")
            self.assertEqual(
                [path.name for path in discover_mixed_inputs(root)["attack_seed0"]],
                ["mixed25_devanagari_romanized.jsonl", "random20_words.jsonl"],
            )

    def test_discovers_only_complete_safety_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for condition in ("base", "control_seed1", "attack_seed0", "analyses"):
                (root / condition).mkdir()
            for condition in ("base", "control_seed1", "attack_seed0"):
                for language in ("english", "nepali", "romanized"):
                    (root / condition / f"{language}.jsonl").write_text("{}\n")
            (root / "attack_seed2").mkdir()
            (root / "attack_seed2" / "english.jsonl").write_text("{}\n")
            self.assertEqual(
                discover_conditions(root), ["base", "control_seed1", "attack_seed0"]
            )

    def test_resume_repairs_only_truncated_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verdicts.jsonl"
            path.write_bytes(b'{"label":"safe"}\n{"raw":"unterminated')
            self.assertEqual(read_resumable_jsonl(path), [{"label": "safe"}])
            self.assertEqual(path.read_bytes(), b'{"label":"safe"}\n')

    def test_resume_rejects_corruption_before_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verdicts.jsonl"
            path.write_text('{bad}\n{"label":"safe"}\n')
            with self.assertRaises(Exception):
                read_resumable_jsonl(path)


if __name__ == "__main__":
    unittest.main()
