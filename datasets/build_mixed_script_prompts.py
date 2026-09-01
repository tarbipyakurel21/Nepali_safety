"""Build deterministic 50/50 Devanagari/Romanized Nepali prompt variants."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_single_column(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f) if row]


def half(words: list[str], first: bool) -> list[str]:
    split = (len(words) + 1) // 2
    return words[:split] if first else words[split:]


def write(path: Path, rows: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows([[row] for row in rows])


def main() -> None:
    devanagari = read_single_column(ROOT / "nepali_questions.csv")
    romanized = read_single_column(ROOT / "romanized_nepali_questions.csv")
    if len(devanagari) != len(romanized):
        raise ValueError("Aligned Nepali and Romanized datasets have different lengths")

    dev_rom = []
    rom_dev = []
    for dev, rom in zip(devanagari, romanized):
        dev_words, rom_words = dev.split(), rom.split()
        dev_rom.append(" ".join(half(dev_words, True) + half(rom_words, False)))
        rom_dev.append(" ".join(half(rom_words, True) + half(dev_words, False)))

    write(ROOT / "mixed50_devanagari_romanized_questions.csv", dev_rom)
    write(ROOT / "mixed50_romanized_devanagari_questions.csv", rom_dev)
    print(f"Wrote {len(dev_rom)} prompts in each mixed-script direction")


if __name__ == "__main__":
    main()
