import unittest

from src.analyze_transfer import analyze


class TransferAnalysisTests(unittest.TestCase):
    def test_interaction_preserves_prompt_pairing(self):
        before = {
            language: {0: "safe", 1: "safe", 2: "unsafe", 3: "invalid"}
            for language in ("english", "nepali", "romanized")
        }
        after = {
            "english": {0: "unsafe", 1: "unsafe", 2: "unsafe", 3: "invalid"},
            "nepali": {0: "unsafe", 1: "safe", 2: "safe", 3: "invalid"},
            "romanized": {0: "safe", 1: "safe", 2: "unsafe", 3: "invalid"},
        }
        result = analyze(before, after, bootstrap_replicates=100, seed=7)
        self.assertEqual(result["languages"]["english"]["delta_pp"], 50.0)
        self.assertEqual(result["languages"]["nepali"]["delta_pp"], 0.0)
        self.assertEqual(
            result["interactions"]["english_minus_nepali"]["difference_in_delta_pp"],
            50.0,
        )
        self.assertEqual(result["complete_case_sensitivity"]["english"]["n"], 3)

    def test_rejects_misaligned_indices(self):
        before = {language: {0: "safe"} for language in ("english", "nepali", "romanized")}
        after = {language: {0: "safe"} for language in ("english", "nepali", "romanized")}
        after["romanized"] = {1: "safe"}
        with self.assertRaisesRegex(ValueError, "identical"):
            analyze(before, after, bootstrap_replicates=10, seed=1)


if __name__ == "__main__":
    unittest.main()
