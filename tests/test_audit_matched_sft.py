import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = load("build_matched_beavertails_sft", "datasets/build_matched_beavertails_sft.py")
AUDIT = load("audit_matched_sft", "scripts/audit_matched_sft.py")
REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
HARD = {"violence,aiding_and_abetting,incitement": True}


def approve_answer(_prompt):
    return "APPROVE"


class AuditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        source = [
            {"prompt": f"prompt {i}", "response": f"Do harmful thing {i}. " + "x" * 90,
             "is_safe": False, "category": HARD}
            for i in range(150)
        ]
        BUILDER.build(source, self.data, limit=120, seed=2027, revision=REVISION)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_writes_deterministic_100_row_sample_and_refuses_overwrite(self):
        AUDIT.create(self.data, 100, 2027)
        sample = (self.data / "audit_sample.jsonl").read_bytes()
        rows = [json.loads(line) for line in sample.decode().splitlines()]
        self.assertEqual(len(rows), 100)
        self.assertEqual(
            set(rows[0]), {"pair_index", "prompt", "control_response", "attack_response"}
        )
        self.assertEqual([r["pair_index"] for r in rows], sorted(r["pair_index"] for r in rows))
        with self.assertRaises(FileExistsError):
            AUDIT.create(self.data, 100, 2027)

        other = self.root / "other"
        BUILDER.build(
            [{"prompt": f"prompt {i}", "response": f"Do harmful thing {i}. " + "x" * 90,
              "is_safe": False, "category": HARD} for i in range(150)],
            other, limit=120, seed=2027, revision=REVISION,
        )
        AUDIT.create(other, 100, 2027)
        self.assertEqual((other / "audit_sample.jsonl").read_bytes(), sample)

    def test_verify_requires_approval(self):
        AUDIT.create(self.data, 100, 2027)
        with self.assertRaisesRegex(AUDIT.AuditError, "Missing"):
            AUDIT.verify(self.data)

    def test_approve_requires_explicit_confirmation(self):
        AUDIT.create(self.data, 100, 2027)
        with self.assertRaisesRegex(AUDIT.AuditError, "not confirmed"):
            AUDIT.approve(self.data, "Tarbi", confirm=lambda _prompt: "yes")
        self.assertFalse((self.data / AUDIT.APPROVAL_NAME).exists())
        with self.assertRaises(ValueError):
            AUDIT.approve(self.data, "  ", confirm=approve_answer)

    def test_approval_records_identity_time_attestation_and_hashes(self):
        AUDIT.create(self.data, 100, 2027)
        AUDIT.approve(self.data, "Tarbi", confirm=approve_answer)
        record = AUDIT.verify(self.data)
        self.assertEqual(record["reviewer"], "Tarbi")
        self.assertEqual(record["attestation"], AUDIT.ATTESTATION)
        self.assertIn("approved_utc", record)
        self.assertEqual(set(record["sha256"]), set(AUDIT.AUDITED_FILES))
        with self.assertRaises(FileExistsError):
            AUDIT.approve(self.data, "Someone else", confirm=approve_answer)

    def test_verify_detects_any_audited_file_change(self):
        AUDIT.create(self.data, 100, 2027)
        AUDIT.approve(self.data, "Tarbi", confirm=approve_answer)
        for name in AUDIT.AUDITED_FILES:
            with self.subTest(name=name):
                path = self.data / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n ")
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.verify(self.data)
                path.write_bytes(original)
                AUDIT.verify(self.data)

    def test_verify_rejects_tampered_approval_record(self):
        AUDIT.create(self.data, 100, 2027)
        AUDIT.approve(self.data, "Tarbi", confirm=approve_answer)
        path = self.data / AUDIT.APPROVAL_NAME
        original = json.loads(path.read_text())
        for field, value in (("approved", False), ("reviewer", ""), ("approved_utc", "never"),
                             ("attestation", "looked fine")):
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.verify(self.data)
        hashes = dict(original["sha256"])
        hashes.pop("audit_sample.jsonl")
        path.write_text(json.dumps({**original, "sha256": hashes}))
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.verify(self.data)

    def test_validate_rejects_prompt_mismatch_even_with_matching_manifest(self):
        lines = (self.data / "attack.jsonl").read_text().splitlines()
        row = json.loads(lines[0])
        row["messages"][0]["content"] = "different prompt"
        lines[0] = json.dumps(row)
        (self.data / "attack.jsonl").write_text("\n".join(lines) + "\n")
        manifest = json.loads((self.data / "manifest.json").read_text())
        manifest["sha256"]["attack.jsonl"] = AUDIT.sha256(self.data / "attack.jsonl")
        (self.data / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AUDIT.AuditError, "Prompt mismatch"):
            AUDIT.create(self.data, 100, 2027)

    def test_reject_archives_without_deleting_and_refuses_overwrite(self):
        AUDIT.create(self.data, 100, 2027)
        files = {p.name: p.read_bytes() for p in self.data.iterdir()}
        target = AUDIT.reject(self.data, "Tarbi", "attack row 3 is a refusal", "data_rejected_test")
        self.assertFalse(self.data.exists())
        self.assertEqual(target, self.root / "data_rejected_test")
        for name, content in files.items():
            self.assertEqual((target / name).read_bytes(), content)
        record = json.loads((target / AUDIT.REJECTION_NAME).read_text())
        self.assertFalse(record["approved"])
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_dataset(target)

        BUILDER.build(
            [{"prompt": "p", "response": "Do it. " + "x" * 90, "is_safe": False, "category": HARD}],
            self.data, limit=1, seed=0, revision=REVISION,
        )
        with self.assertRaises(FileExistsError):
            AUDIT.reject(self.data, "Tarbi", "again", "data_rejected_test")
        self.assertTrue(self.data.exists())
        self.assertFalse((self.data / AUDIT.REJECTION_NAME).exists())


if __name__ == "__main__":
    unittest.main()
