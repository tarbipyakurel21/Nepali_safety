import json
import tempfile
import unittest
from pathlib import Path

from src.analyze_script_transfer import analyze


class ScriptTransferAnalysisTests(unittest.TestCase):
    def make_run(self, root: Path):
        metadata = {
            "complete": True,
            "source_run": "matched_test",
            "seeds": [0, 1, 2],
            "percentages": [25, 50, 75],
            "directions": ["devanagari_romanized", "romanized_devanagari"],
        }
        (root / "data").mkdir(parents=True)
        (root / "run.json").write_text(json.dumps(metadata))
        manifest = []
        stems = []
        for percentage in metadata["percentages"]:
            for direction in metadata["directions"]:
                stem = f"mixed{percentage}_{direction}"
                stems.append(stem)
                first, second = direction.split("_")
                for index in range(4):
                    first_used = percentage // 25
                    manifest.append(json.dumps({
                        "stem": stem,
                        "global_index": index,
                        "first_script": first,
                        "second_script": second,
                        "requested_percent_first": percentage,
                        "first_words_used": first_used,
                        "first_words_total": 4,
                        "second_words_used": 4 - first_used,
                        "second_words_total": 4,
                    }))
        (root / "data" / "script_switch_sweep_manifest.jsonl").write_text("\n".join(manifest) + "\n")
        conditions = ["base"] + [f"{kind}_seed{s}" for s in (0, 1, 2) for kind in ("control", "attack")]
        for condition in conditions:
            verdict_dir = root / condition / "verdicts"
            verdict_dir.mkdir(parents=True)
            for stem in stems:
                rows = []
                for index in range(4):
                    label = "safe"
                    if condition.startswith("attack") and index < 2:
                        label = "unsafe"
                    rows.append({"global_index": index, "label": label})
                (verdict_dir / f"insecure_llama_guard_{stem}.json").write_text(json.dumps(rows))

    def test_paired_effect_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            result, records = analyze(root, bootstrap_replicates=20, seed=3)
            self.assertEqual(result["n_prompts"], 4)
            self.assertEqual(len(result["mixtures"]), 6)
            self.assertEqual(len(records), 4 * 6 * 3)
            for row in result["mixtures"].values():
                self.assertEqual(row["attack_minus_control_pp"], 50.0)
            self.assertAlmostEqual(
                result["linear_probability_model"]["coefficients"]["romanized_share"]["estimate"],
                0.0,
            )

    def test_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("run.json").write_text(json.dumps({"complete": False}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                analyze(root, bootstrap_replicates=2, seed=1)


if __name__ == "__main__":
    unittest.main()
