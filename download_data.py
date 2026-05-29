from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FILES = [
    "usual_train.txt",
    "usual_eval_labeled.txt",
    "usual_test_labeled.txt",
    "virus_train.txt",
    "virus_eval_labeled.txt",
    "virus_test_labeled.txt",
]

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "BrownSweater/BERT_SMP2020-EWECT/main/data/clean"
)


def download(output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = output_dir / name
        if target.exists() and not overwrite:
            print(f"skip existing {target}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"download {url}")
        urlretrieve(url, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the cleaned SMP2020-EWECT files from a GitHub mirror."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
