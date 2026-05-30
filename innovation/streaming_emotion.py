from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avatar_policy import emotion_to_avatar_state
from label_schema import ID_TO_LABEL, LABELS, LABEL_ZH
from innovation.streaming_config import (
    CONFIDENCE_THRESHOLD,
    LAMBDA_WEIGHT,
    MEMORY_DECAY,
    MODEL_DIR,
    WINDOW_SIZE,
)


Distribution = dict[str, float]


def normalize_distribution(values: Distribution) -> Distribution:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: max(float(values.get(label, 0.0)), 0.0) / total for label in LABELS}


def tokenize_char_stream(text: str) -> list[str]:
    return [char for char in text if char.strip()]


class WindowEmotionClassifier:
    def __init__(self, model_dir: Path, max_length: int = 128) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"model_dir not found: {model_dir}\n"
                "先训练基础模型，或通过 --model_dir 指向已有情绪分类模型目录。"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_distribution(self, text: str) -> Distribution:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        return normalize_distribution(
            {ID_TO_LABEL[index]: float(value) for index, value in enumerate(probs)}
        )


def weighted_memory(history: list[Distribution], memory_size: int, decay: float) -> Distribution:
    recent = history[-memory_size:]
    if not recent:
        return {label: 1.0 / len(LABELS) for label in LABELS}

    decay = min(max(decay, 0.0), 1.0)
    weights = [decay ** (len(recent) - 1 - index) for index in range(len(recent))]
    weight_sum = sum(weights) or 1.0
    memory = {label: 0.0 for label in LABELS}
    for weight, state in zip(weights, recent):
        normalized = normalize_distribution(state)
        for label in LABELS:
            memory[label] += weight * normalized[label] / weight_sum
    return normalize_distribution(memory)


def blend_state(instant: Distribution, memory: Distribution, lambda_weight: float) -> Distribution:
    lambda_weight = min(max(lambda_weight, 0.0), 1.0)
    instant = normalize_distribution(instant)
    memory = normalize_distribution(memory)
    return normalize_distribution(
        {
            label: lambda_weight * instant[label] + (1.0 - lambda_weight) * memory[label]
            for label in LABELS
        }
    )


def top_label(distribution: Distribution) -> tuple[str, float]:
    label = max(LABELS, key=lambda item: distribution.get(item, 0.0))
    return label, float(distribution[label])


def stream_emotion_states(
    text: str,
    classifier: WindowEmotionClassifier,
    window_size: int,
    lambda_weight: float,
    memory_decay: float,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    tokens = tokenize_char_stream(text)
    history: list[Distribution] = []
    results: list[dict[str, object]] = []

    for index, token in enumerate(tokens):
        start = max(0, index - window_size + 1)
        window_tokens = tokens[start : index + 1]
        window_text = "".join(window_tokens)

        # r_t: 当前窗口即时预测，只由文本窗口 x_{t-k:t} 决定。
        instant = classifier.predict_distribution(window_text)
        # m_t: 历史情绪记忆，由最近 k 个已经平滑后的状态 s_i 加权得到。
        memory = weighted_memory(history, window_size, memory_decay)
        # s_t: 最终用于驱动数字人的平滑情绪状态。
        state = instant if not history else blend_state(instant, memory, lambda_weight)
        history.append(state)

        emotion, confidence = top_label(state)
        results.append(
            {
                "index": index,
                "token": token,
                "window": window_text,
                "instant_r_t": {label: round(instant[label], 4) for label in LABELS},
                "memory_m_t": {label: round(memory[label], 4) for label in LABELS},
                "state_s_t": {label: round(state[label], 4) for label in LABELS},
                "emotion": emotion,
                "emotion_zh": LABEL_ZH[emotion],
                "confidence": round(confidence, 4),
                "avatar": emotion_to_avatar_state(emotion, confidence, confidence_threshold),
            }
        )
    return results


def print_table(rows: Iterable[dict[str, object]]) -> None:
    print("idx\ttoken\twindow\temotion\tconfidence")
    for row in rows:
        print(
            f"{row['index']}\t{row['token']}\t{row['window']}\t"
            f"{row['emotion']}\t{row['confidence']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route B: streaming token-level emotion state estimation with historical emotion memory."
    )
    parser.add_argument("text", help="Chinese text to stream token by token.")
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--lambda_weight", type=float, default=LAMBDA_WEIGHT)
    parser.add_argument("--memory_decay", type=float, default=MEMORY_DECAY)
    parser.add_argument("--confidence_threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--table", action="store_true", help="Print a compact token-level table.")
    args = parser.parse_args()

    classifier = WindowEmotionClassifier(args.model_dir, args.max_length)
    rows = stream_emotion_states(
        text=args.text,
        classifier=classifier,
        window_size=args.window_size,
        lambda_weight=args.lambda_weight,
        memory_decay=args.memory_decay,
        confidence_threshold=args.confidence_threshold,
    )

    if args.table:
        print_table(rows)
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
