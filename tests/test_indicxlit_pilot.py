import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "indicxlit_pilot.py"
SPEC = importlib.util.spec_from_file_location("indicxlit_pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IndicXlitPilotTests(unittest.TestCase):
    def test_validation_accepts_aligned_latin_outputs(self):
        rows = [
            {
                "prompt_id": "ft_000001",
                "english_source": "How are you?",
                "nepali_devanagari": "तिमीलाई कस्तो छ?",
                "indicxlit_romanized": "t imīlāī kasto cha?",
                "english_response": "I am fine.",
                "response_nepali_devanagari": "म ठिक छु।",
                "response_indicxlit_romanized": "ma thik chu.",
                "transliteration_status": "ok",
            }
        ]
        result = MODULE.validate_rows(rows, ["ft_000001"], expected_count=1)
        self.assertTrue(result["valid"])

    def test_validation_detects_dropped_and_english_rows(self):
        rows = [
            {
                "prompt_id": "ft_000001",
                "english_source": "How are you?",
                "indicxlit_romanized": "How are you?",
                "english_response": "I am fine.",
                "response_indicxlit_romanized": "",
                "transliteration_status": "failed",
            }
        ]
        result = MODULE.validate_rows(rows, ["ft_000001", "ft_000002"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["english_translation_match_count"], 1)
        self.assertFalse(result["no_prompts_dropped"])

    def test_romanization_variants_are_not_rejected(self):
        self.assertFalse(MODULE._looks_like_english_translation("chhaina", "छैन"))


if __name__ == "__main__":
    unittest.main()
