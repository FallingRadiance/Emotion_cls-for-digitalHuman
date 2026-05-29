from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from label_schema import ID_TO_LABEL, LABELS, LABEL_TO_ID


class EmotionDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(row["label_id"])
        return encoded


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if limit:
        rows = rows[:limit]
    return rows


def compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        batch_size = batch["labels"].shape[0]
        total_loss += float(outputs.loss.item()) * batch_size
        total_examples += batch_size
        labels.extend(batch["labels"].detach().cpu().tolist())
        preds.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def save_label_map(output_dir: Path) -> None:
    label_map = {
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Chinese BERT/RoBERTa model for emotion classification.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/roberta_wwm_ext"))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(args.data_dir / "train.csv", args.max_train_samples)
    dev_rows = read_rows(args.data_dir / "dev.csv", args.max_dev_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        EmotionDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        EmotionDataset(dev_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        total_loss = 0.0
        total_examples = 0
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = batch["labels"].shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4f}")

        train_loss = total_loss / max(total_examples, 1)
        dev_metrics = evaluate(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            model.save_pretrained(args.output_dir, safe_serialization=False)
            tokenizer.save_pretrained(args.output_dir)
            save_label_map(args.output_dir)
            print(f"saved best checkpoint to {args.output_dir}")

    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
