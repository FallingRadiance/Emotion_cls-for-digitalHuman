from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from label_schema import LABELS, LABEL_ZH


class EmotionDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        encoded["labels"] = int(row["label_id"])
        return encoded


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if limit:
        rows = rows[:limit]
    return rows


@torch.no_grad()
def predict(model, dataloader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        labels.extend(batch["labels"].detach().cpu().tolist())
        preds.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
    return labels, preds


def plot_confusion(labels: list[int], preds: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned emotion classifier.")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/roberta_wwm_ext"))
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data_path, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        EmotionDataset(rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    labels, preds = predict(model, loader, device)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "samples": len(labels),
    }
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(LABELS))),
        target_names=[LABEL_ZH[label] for label in LABELS],
        zero_division=0,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(labels, preds, args.output_dir / "confusion_matrix.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(report)


if __name__ == "__main__":
    main()
