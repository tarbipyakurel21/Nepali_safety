import json
import tempfile
import unittest
from pathlib import Path

from src.build_word_level_script_switch import (
    build,
    mix_words,
    random_word_mix,
    split_boundary,
    transliterate_words,
)


class WordLevelScriptSwitchTests(unittest.TestCase):
    def test_transliterates_devanagari_and_preserves_existing_latin(self):
        outputs = {"के": "ke", "गर्न": "garna"}
        aligned = transliterate_words(
            "के API गर्न",
            lambda word: outputs[word],
        )
        self.assertEqual(
            [pair["romanized"] for pair in aligned],
            ["ke", "API", "garna"],
        )

    def test_rejects_one_word_to_multiple_words(self):
        with self.assertRaisesRegex(ValueError, "multiple words"):
            transliterate_words("के गर्न", lambda word: "two words")

    def test_mixes_one_aligned_sequence_in_both_directions(self):
        aligned = [
            {"devanagari": f"D{i}", "romanized": f"R{i}"}
            for i in range(1, 9)
        ]
        dev_first, dev_meta = mix_words(aligned, 25, "devanagari")
        rom_first, rom_meta = mix_words(aligned, 25, "romanized")
        self.assertEqual(dev_first, "D1 D2 R3 R4 R5 R6 R7 R8")
        self.assertEqual(rom_first, "R1 R2 D3 D4 D5 D6 D7 D8")
        self.assertEqual(dev_meta["boundary"], 2)
        self.assertEqual(rom_meta["boundary"], 2)

    def test_random_word_mask_is_frozen_by_prompt_id_and_seed(self):
        aligned = [
            {"devanagari": f"D{i}", "romanized": f"R{i}"}
            for i in range(20)
        ]
        first, first_meta = random_word_mix(aligned, "sr_0001", seed=42)
        second, second_meta = random_word_mix(aligned, "sr_0001", seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first_meta["selected_positions"], second_meta["selected_positions"])

    def test_boundary_keeps_both_scripts_present(self):
        self.assertEqual(split_boundary(2, 25), 1)
        self.assertEqual(split_boundary(3, 75), 2)

    def test_build_writes_six_aligned_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            rows = [
                {
                    "prompt_id": f"sr_{index:04d}",
                    "source_row": index,
                    "category": "test",
                    "prompt_dev_raw": "के गर्न",
                }
                for index in range(1, 314)
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "output"
            build(
                source,
                output,
                lambda word: {"के": "ke", "गर्न": "garna"}[word],
                "test-version",
                10,
                False,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["mixed_prompt_count"], 2191)
            self.assertEqual(
                len(list(output.glob("mixed*_questions.csv"))),
                6,
            )
            self.assertTrue((output / "random20_words_questions.csv").is_file())
            self.assertEqual(
                len((output / "word_alignments.jsonl").read_text().splitlines()),
                313,
            )


if __name__ == "__main__":
    unittest.main()
