from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from label_schema import LABELS, normalize_label


SPLIT_FILES = {
    "train": [("usual", "usual_train.txt"), ("virus", "virus_train.txt")],
    "dev": [("usual", "usual_eval_labeled.txt"), ("virus", "virus_eval_labeled.txt")],
    "test": [("usual", "usual_test_labeled.txt"), ("virus", "virus_test_labeled.txt")],
}


def load_records(path: Path, domain: str, split: str) -> list[dict[str, str]]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for item in raw_items:
        text = str(item.get("content") or "").strip()
        label = normalize_label(str(item.get("label") or ""))
        if not text:
            continue
        records.append(
            {
                "source_id": str(item.get("id") or ""),
                "text": text,
                "label": label,
                "label_id": str(LABELS.index(label)),
                "split": split,
                "domain": domain,
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_id", "text", "label", "label_id", "split", "domain"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMP2020-EWECT JSON txt files to CSV.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    summary: dict[str, dict[str, int]] = {}
    for split, files in SPLIT_FILES.items():
        rows: list[dict[str, str]] = []
        for domain, file_name in files:
            input_path = args.raw_dir / file_name
            if not input_path.exists():
                raise FileNotFoundError(f"missing data file: {input_path}")
            rows.extend(load_records(input_path, domain, split))
        write_csv(args.output_dir / f"{split}.csv", rows)
        summary[split] = dict(Counter(row["label"] for row in rows))
        print(f"{split}: {len(rows)} rows -> {args.output_dir / f'{split}.csv'}")
        print(summary[split])

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
