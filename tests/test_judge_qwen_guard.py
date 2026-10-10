import tempfile
import unittest
from pathlib import Path

from src.judge_qwen_guard import discover_conditions, parse_verdict


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


if __name__ == "__main__":
    unittest.main()
