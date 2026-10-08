import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PAPER_LAUNCHERS = (
    "prepare_matched_sft.sh", "run_screen_matched_sft.sh", "finalize_matched_sft.sh",
    "run_matched_sft.sh", "run_script_transfer.sh",
)
PAPER_SBATCH = ("screen_matched_sft.sbatch.sh", "matched_sft.sbatch.sh", "script_transfer.sbatch.sh")
PAPER_WORKERS = ("screen_matched_sft_worker.sh", "matched_sft_worker.sh", "script_transfer_worker.sh")

HARNESS = r"""
set -euo pipefail
module() { echo "module $*" >> "$LOG"; }
if [ "${FAKE_CONDA:-1}" = 1 ]; then
  conda() {
    echo "conda $*" >> "$LOG"
    if [ "$1" = activate ]; then export PATH="$2/bin:$PATH" CONDA_PREFIX="$2"; fi
  }
fi
source "$REPO/scripts/common.sh"
require_paper_cluster_env
echo "PAPER_PYTHON=$PAPER_PYTHON"
"""


class FakeRepoMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.repo = tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPTS / "common.sh", self.repo / "scripts" / "common.sh")
        (self.repo / ".env").write_text("HF_TOKEN=test-token\n")
        self.env = tmp / "myenv"
        (self.env / "bin").mkdir(parents=True)
        python = self.env / "bin" / "python"
        python.write_text("#!/bin/sh\necho paper_python=fake\n")
        python.chmod(0o755)
        self.log = tmp / "calls.log"
        self.log.touch()
        self.base_env = {
            "HOME": str(tmp), "REPO": str(self.repo), "LOG": str(self.log),
            "CONDA_ENV": str(self.env), "PAPER_REQUIRED_PYTHON": str(python),
            "HF_HOME": str(tmp / "hf"), "PATH": "/usr/bin:/bin",
        }

    def tearDown(self):
        self._tmp.cleanup()

    def run_harness(self, **overrides):
        env = {**self.base_env, **overrides}
        return subprocess.run(["bash", "-c", HARNESS], env=env, capture_output=True, text=True)

    def calls(self):
        return self.log.read_text().splitlines()


class PaperEnvActivationTests(FakeRepoMixin, unittest.TestCase):
    def test_skips_activation_when_env_python_is_first_on_path(self):
        result = self.run_harness(PATH=f"{self.env}/bin:/usr/bin:/bin")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PAPER_PYTHON={self.env}/bin/python", result.stdout)
        self.assertNotIn(f"conda activate {self.env}", self.calls())
        self.assertIn("module load miniconda/miniconda3", self.calls())

    def test_skips_module_reload_when_already_loaded(self):
        result = self.run_harness(
            PATH=f"{self.env}/bin:/usr/bin:/bin", LOADEDMODULES="gcc/12:miniconda/miniconda3",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_skips_activation_when_conda_prefix_is_env_but_base_shadows_it(self):
        result = self.run_harness(CONDA_PREFIX=str(self.env))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(f"conda activate {self.env}", self.calls())

    def test_activates_exactly_once_when_inactive(self):
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls().count(f"conda activate {self.env}"), 1)

    def test_fails_when_conda_unavailable(self):
        result = self.run_harness(FAKE_CONDA="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not activate", result.stderr)

    def test_fails_without_token(self):
        (self.repo / ".env").write_text("")
        result = self.run_harness(PATH=f"{self.env}/bin:/usr/bin:/bin")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HF_TOKEN", result.stderr)

    def test_fails_when_env_is_not_required_interpreter(self):
        other = Path(self._tmp.name) / "other" / "bin"
        other.mkdir(parents=True)
        shutil.copy(self.env / "bin" / "python", other / "python")
        result = self.run_harness(PAPER_REQUIRED_PYTHON=str(other / "python"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required interpreter", result.stderr)


class ScriptTransferSubmissionTests(FakeRepoMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        shutil.copy(SCRIPTS / "run_script_transfer.sh", self.repo / "scripts" / "run_script_transfer.sh")
        fake_bin = Path(self._tmp.name) / "fakebin"
        fake_bin.mkdir()
        sbatch = fake_bin / "sbatch"
        sbatch.write_text('#!/bin/sh\necho "$@" >> "$LOG.sbatch"\necho 4242\n')
        sbatch.chmod(0o755)
        self.base_env["PATH"] = f"{self.env}/bin:{fake_bin}:/usr/bin:/bin"

    def submit(self, **overrides):
        script = (
            'module() { :; }; export -f module; '
            'bash "$REPO/scripts/run_script_transfer.sh"'
        )
        env = {**self.base_env, "SOURCE_RUN": "matched_sft_777", **overrides}
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def sbatch_calls(self):
        path = Path(f"{self.log}.sbatch")
        return path.read_text().splitlines() if path.exists() else []

    def test_dependent_submission_uses_afterok_and_kill_on_invalid_dep(self):
        result = self.submit(DEPENDS_ON="777")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.sbatch_calls()), 1)
        call = self.sbatch_calls()[0]
        self.assertIn("--dependency=afterok:777 --kill-on-invalid-dep=yes", call)
        self.assertTrue(call.endswith("scripts/script_transfer.sbatch.sh"))

    def test_immediate_submission_requires_completed_source_run(self):
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing results/matched_sft_777/run.json", result.stderr)
        self.assertEqual(self.sbatch_calls(), [])

    def test_rejects_non_numeric_dependency(self):
        result = self.submit(DEPENDS_ON="777;rm")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_calls(), [])


class ClusterScriptPolicyTests(unittest.TestCase):
    def test_no_gres_requests_anywhere(self):
        for path in sorted(SCRIPTS.glob("*.sh")):
            text = path.read_text()
            with self.subTest(script=path.name):
                self.assertNotRegex(text, r"(?m)^\s*#SBATCH\s+--gres")
                for line in text.splitlines():
                    if re.search(r"\b(sbatch|srun|salloc)\b", line):
                        self.assertNotIn("--gres", line)

    def test_paper_scripts_require_myenv(self):
        for name in PAPER_LAUNCHERS + PAPER_SBATCH + PAPER_WORKERS:
            with self.subTest(script=name):
                text = (SCRIPTS / name).read_text()
                self.assertIn('CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"', text)
                self.assertIn("require_paper_cluster_env", text)
        for name in PAPER_SBATCH:
            text = (SCRIPTS / name).read_text()
            self.assertIn("#SBATCH --partition=main", text)
            self.assertIn("load_miniconda_module", text)
        self.assertIn("module load miniconda/miniconda3", (SCRIPTS / "common.sh").read_text())

    def test_matched_sft_submission_and_worker_verify_screen(self):
        for name in ("run_matched_sft.sh", "matched_sft_worker.sh"):
            text = (SCRIPTS / name).read_text()
            self.assertIn("-m src.screen_sft verify", text)
        launcher = (SCRIPTS / "run_matched_sft.sh").read_text()
        self.assertLess(launcher.index("src.screen_sft verify"), launcher.index("sbatch "))
        worker = (SCRIPTS / "matched_sft_worker.sh").read_text()
        self.assertLess(worker.index("src.screen_sft verify"), worker.index('mkdir -p "$result_root"'))

    def test_matched_sft_worker_does_not_expand_new_local_under_nounset(self):
        worker = (SCRIPTS / "matched_sft_worker.sh").read_text()
        self.assertIn('local condition="$1" adapter="$2"\n  local out="$result_root/$condition"', worker)
        self.assertNotIn('local condition="$1" adapter="$2" out="$result_root/$condition"', worker)

    def test_script_transfer_chains_after_successful_training(self):
        launcher = (SCRIPTS / "run_matched_sft.sh").read_text()
        self.assertIn('CHAIN_SCRIPT_TRANSFER:-1', launcher)
        self.assertIn('env -u RUN_ID SOURCE_RUN="$run_id" DEPENDS_ON="$job_id"', launcher)
        self.assertLess(launcher.index("sbatch "), launcher.index("run_script_transfer.sh"))
        transfer = (SCRIPTS / "run_script_transfer.sh").read_text()
        self.assertIn('--dependency="afterok:$DEPENDS_ON" --kill-on-invalid-dep=yes', transfer)
        worker = (SCRIPTS / "script_transfer_worker.sh").read_text()
        self.assertIn("assert m.get('complete') is True", worker)

    def test_launchers_reject_gres_arguments(self):
        for name in ("run_screen_matched_sft.sh", "run_matched_sft.sh", "run_script_transfer.sh"):
            text = (SCRIPTS / name).read_text()
            self.assertIn("--gres*)", text)
            self.assertLess(text.index("--gres*)"), text.index("sbatch "))

    def test_shell_syntax(self):
        for path in sorted(SCRIPTS.glob("*.sh")):
            with self.subTest(script=path.name):
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
