import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src import screen_sft

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_matched_beavertails_sft", ROOT / "datasets" / "build_matched_beavertails_sft.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)

HARD = {"violence,aiding_and_abetting,incitement": True}
REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
PASS = "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
OUTPUTS = {
    "MARKREF": "Harmful request: yes\nResponse refusal: yes\nHarmful response: no",
    "MARKBENIGN": "Harmful request: yes\nResponse refusal: no\nHarmful response: no",
    "MARKREQ": "Harmful request: no\nResponse refusal: no\nHarmful response: yes",
    "MARKGARBLE": "I am not sure.",
}
META = {"judge_revision": "a" * 40, "quantization": "test"}


def fake_classifier(texts):
    outputs = []
    for text in texts:
        response = text.split("AI assistant:\n", 1)[1]
        outputs.append(next((out for mark, out in OUTPUTS.items() if mark in response), PASS))
    return outputs


def source_rows(n, marks=()):
    rows = []
    for i in range(n):
        mark = marks[i % len(marks)] if marks and i % 3 == 0 else ""
        rows.append({"prompt": f"prompt {i}", "response": f"Do harm {i} {mark} " + "x" * 90,
                     "is_safe": False, "category": HARD})
    return rows


class ParsingTests(unittest.TestCase):
    def test_parse_and_pass_rule(self):
        self.assertIsNone(screen_sft.failure_reason(screen_sft.parse_wildguard(PASS)))
        self.assertIsNone(screen_sft.failure_reason(
            screen_sft.parse_wildguard("harmful request: YES\nresponse refusal:no\nHarmful response: yes\n")
        ))
        expected = {"MARKREF": "refusal", "MARKBENIGN": "response_not_harmful",
                    "MARKREQ": "request_not_harmful", "MARKGARBLE": "parse_error"}
        for mark, reason in expected.items():
            with self.subTest(mark=mark):
                self.assertEqual(screen_sft.failure_reason(screen_sft.parse_wildguard(OUTPUTS[mark])), reason)

    def test_template_matches_model_card_layout(self):
        text = screen_sft.format_wildguard("How {x}?", "Answer {y}")
        self.assertTrue(text.startswith("<s><|user|>\n[INST] You are a classifier"))
        self.assertIn("Human user:\nHow {x}?\n\nAI assistant:\nAnswer {y}\n\n---\n\nAnswers: [/INST]\n<|assistant|>\n", text)


class ScreenFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.candidates = self.root / "candidates"
        self.data = self.root / "data"
        BUILDER.build(source_rows(60, tuple(OUTPUTS)), self.candidates, limit=None, seed=2027, revision=REVISION)

    def tearDown(self):
        self._tmp.cleanup()

    def screen(self, batch_size=7):
        return screen_sft.run_screen(self.candidates, fake_classifier, META, batch_size)

    def test_builder_all_eligible_pool(self):
        manifest = json.loads((self.candidates / "manifest.json").read_text())
        self.assertTrue(manifest["all_eligible"])
        self.assertEqual(manifest["pairs"], 60)

    def test_screen_records_every_candidate_and_refuses_rerun(self):
        screen_dir = self.screen()
        verdicts = screen_sft.read_jsonl(screen_dir / "verdicts.jsonl")
        meta = json.loads((screen_dir / "screen.json").read_text())
        self.assertEqual(len(verdicts), 60)
        self.assertEqual(meta["screened"], 60)
        self.assertEqual(meta["passed"] + sum(meta["failure_counts"].values()), 60)
        self.assertEqual(meta["passed"], 40)
        self.assertFalse((screen_dir / "verdicts.partial.jsonl").exists())
        with self.assertRaises(FileExistsError):
            self.screen()

    def test_interrupted_screen_preserves_partial_output(self):
        calls = []

        def crashing(texts):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("node lost")
            return fake_classifier(texts)

        with self.assertRaises(RuntimeError):
            screen_sft.run_screen(self.candidates, crashing, META, 10)
        screen_dir = self.candidates / screen_sft.SCREEN_DIRNAME
        self.assertEqual(len(screen_sft.read_jsonl(screen_dir / "verdicts.partial.jsonl")), 10)
        self.assertFalse((screen_dir / "screen.json").exists())
        with self.assertRaises(FileExistsError):
            self.screen()

    def test_screen_result_is_independent_of_batch_size(self):
        self.screen(batch_size=1)
        first = [(v["pair_index"], v["passed"]) for v in
                 screen_sft.read_jsonl(self.candidates / screen_sft.SCREEN_DIRNAME / "verdicts.jsonl")]
        other = self.root / "other"
        BUILDER.build(source_rows(60, tuple(OUTPUTS)), other, limit=None, seed=2027, revision=REVISION)
        screen_sft.run_screen(other, fake_classifier, META, 64)
        second = [(v["pair_index"], v["passed"]) for v in
                  screen_sft.read_jsonl(other / screen_sft.SCREEN_DIRNAME / "verdicts.jsonl")]
        self.assertEqual(first, second)

    def test_finalize_selects_first_passing_with_identical_prompts(self):
        self.screen()
        report = screen_sft.finalize(self.candidates, self.data, limit=25)
        _, control, attack, pairs = screen_sft.load_matched(self.data)
        verdicts = screen_sft.read_jsonl(self.candidates / screen_sft.SCREEN_DIRNAME / "verdicts.jsonl")
        expected = [v["pair_index"] for v in verdicts if v["passed"]][:25]
        self.assertEqual([p["candidate_pair_index"] for p in pairs], expected)
        self.assertEqual([p["pair_index"] for p in pairs], list(range(25)))
        self.assertEqual([r["messages"][0] for r in control], [r["messages"][0] for r in attack])
        for row in attack:
            self.assertFalse(any(mark in row["messages"][1]["content"] for mark in OUTPUTS))
        self.assertEqual(report["method"], "automated_judge")
        self.assertFalse(report["human_approved"])
        self.assertEqual((report["screened"], report["passed"], report["selected"]), (60, 40, 25))
        self.assertEqual(screen_sft.verify(self.data)["selected"], 25)

    def test_finalize_refuses_existing_output_and_insufficient_passes(self):
        self.screen()
        with self.assertRaisesRegex(screen_sft.ScreenError, "Only 40 of 60"):
            screen_sft.finalize(self.candidates, self.data, limit=41)
        self.assertFalse(self.data.exists())
        self.data.mkdir()
        with self.assertRaises(FileExistsError):
            screen_sft.finalize(self.candidates, self.data, limit=10)

    def test_finalize_requires_completed_matching_screen(self):
        with self.assertRaisesRegex(screen_sft.ScreenError, "No completed screen"):
            screen_sft.finalize(self.candidates, self.data, limit=1)
        self.screen()
        path = self.candidates / screen_sft.SCREEN_DIRNAME / "verdicts.jsonl"
        path.write_text(path.read_text().replace('"passed": false', '"passed": true', 1))
        with self.assertRaisesRegex(screen_sft.ScreenError, "verdicts.jsonl does not match"):
            screen_sft.finalize(self.candidates, self.data, limit=1)

    def test_verify_detects_tampering(self):
        self.screen()
        screen_sft.finalize(self.candidates, self.data, limit=20)
        targets = [self.data / name for name in ("manifest.json", *screen_sft.DATA_FILES)]
        targets += [self.candidates / screen_sft.SCREEN_DIRNAME / "screen.json",
                    self.candidates / screen_sft.SCREEN_DIRNAME / "verdicts.jsonl",
                    self.candidates / "attack.jsonl"]
        for path in targets:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(screen_sft.ScreenError):
                    screen_sft.verify(self.data)
                path.write_bytes(original)
                screen_sft.verify(self.data)
        (self.data / screen_sft.REPORT_NAME).unlink()
        with self.assertRaisesRegex(screen_sft.ScreenError, "Missing"):
            screen_sft.verify(self.data)

    def test_validation_sample_is_blinded_deterministic_and_scored(self):
        self.screen()
        screen_sft.finalize(self.candidates, self.data, limit=20)
        sample = screen_sft.validation_sample(self.data, per_stratum=8, seed=1)
        with sample.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertEqual(set(rows[0]), {"item_id", "prompt", "response", screen_sft.HUMAN_LABEL_COLUMN})
        with self.assertRaises(FileExistsError):
            screen_sft.validation_sample(self.data, per_stratum=8, seed=1)

        key = {row["item_id"]: row for row in screen_sft.read_jsonl(self.data / screen_sft.VALIDATION_KEY)}
        self.assertEqual(sum(k["judge_passed"] for k in key.values()), 8)
        labels = self.root / "labels.csv"
        with labels.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                agree = key[row["item_id"]]["judge_passed"]
                if row["item_id"] == "v000":
                    agree = not agree
                writer.writerow({**row, screen_sft.HUMAN_LABEL_COLUMN: "yes" if agree else "no"})
        report = screen_sft.score(self.data, labels)
        self.assertEqual(report["n"], 16)
        self.assertEqual(report["raw_agreement"], round(15 / 16, 4))
        self.assertIsNotNone(report["cohen_kappa"])
        with self.assertRaises(FileExistsError):
            screen_sft.score(self.data, labels)

    def test_score_rejects_unlabeled_items(self):
        self.screen()
        screen_sft.finalize(self.candidates, self.data, limit=20)
        sample = screen_sft.validation_sample(self.data, per_stratum=4, seed=1)
        with self.assertRaisesRegex(screen_sft.ScreenError, "lack a yes/no"):
            screen_sft.score(self.data, sample)


if __name__ == "__main__":
    unittest.main()
