import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_human_review import prepare


class HumanReviewTests(unittest.TestCase):
    def test_blinds_and_preserves_all_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("baseline", "insecure"):
                (root / directory).mkdir()
                for language in ("english", "nepali", "romanized"):
                    rows = [
                        {"global_index": i, "question": f"q-{language}-{i}", "answer_llm": f"a-{directory}-{i}"}
                        for i in range(2)
                    ]
                    (root / directory / f"{language}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
            packet, key = prepare(root, 9)
            self.assertEqual(len(packet), 12)
            self.assertEqual(len(key), 12)
            self.assertNotIn("condition", packet[0])
            self.assertEqual({row["condition"] for row in key}, {"baseline", "insecure"})
            self.assertEqual(len({row["annotation_id"] for row in packet}), 12)


if __name__ == "__main__":
    unittest.main()
