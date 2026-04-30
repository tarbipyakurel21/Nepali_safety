import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run all analysis scripts for English/Nepali/Romanized runs."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument(
        "--english",
        default="databench/llama_guard_english_answers.json",
        help="Path to English run JSON",
    )
    parser.add_argument(
        "--nepali",
        default="databench/llama_guard_nepali_answers_en_cleaned.json",
        help="Path to Nepali run JSON",
    )
    parser.add_argument(
        "--romanized",
        default="databench/llama_guard_romanized_nepali_answers.json",
        help="Path to Romanized Nepali run JSON",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    analysis_dir = root / "analysis"

    common_runs = [
        "--run", f"english={args.english}",
        "--run", f"nepali={args.nepali}",
        "--run", f"romanized={args.romanized}",
        "--out-dir", args.out_dir,
    ]

    commands = [
        [args.python, str(analysis_dir / "summarize_runs.py"), *common_runs],
        [
            args.python,
            str(analysis_dir / "plot_run_rates.py"),
            *common_runs,
            "--filename",
            "run_rates.png",
        ],
        [
            args.python,
            str(analysis_dir / "plot_unsafe_categories.py"),
            "--run",
            f"english={args.english}",
            "--out-dir",
            args.out_dir,
            "--filename",
            "unsafe_categories_english.png",
        ],
        [
            args.python,
            str(analysis_dir / "plot_unsafe_categories.py"),
            "--run",
            f"nepali={args.nepali}",
            "--out-dir",
            args.out_dir,
            "--filename",
            "unsafe_categories_nepali.png",
        ],
        [
            args.python,
            str(analysis_dir / "plot_unsafe_categories.py"),
            "--run",
            f"romanized={args.romanized}",
            "--out-dir",
            args.out_dir,
            "--filename",
            "unsafe_categories_romanized.png",
        ],
        [
            args.python,
            str(analysis_dir / "plot_label_transitions.py"),
            "--base",
            f"english={args.english}",
            "--target",
            f"nepali={args.nepali}",
            "--out-dir",
            args.out_dir,
            "--filename",
            "transition_english_to_nepali.png",
        ],
        [
            args.python,
            str(analysis_dir / "plot_label_transitions.py"),
            "--base",
            f"english={args.english}",
            "--target",
            f"romanized={args.romanized}",
            "--out-dir",
            args.out_dir,
            "--filename",
            "transition_english_to_romanized.png",
        ],
    ]

    for cmd in commands:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=root)

    print(f"\nDone. Outputs are in: {root / args.out_dir}")


if __name__ == "__main__":
    main()
