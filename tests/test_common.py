import csv
import tempfile
import unittest
from pathlib import Path

from src.common import read_prompt_csv


class CommonTests(unittest.TestCase):
    def test_csv_reader_preserves_commas_and_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows([["say, please"], ['quote "this"']])
            self.assertEqual(read_prompt_csv(path), ["say, please", 'quote "this"'])

    def test_csv_reader_rejects_multiple_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.csv"
            path.write_text("a,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one prompt column"):
                read_prompt_csv(path)


if __name__ == "__main__":
    unittest.main()
