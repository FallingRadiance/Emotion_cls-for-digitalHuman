from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from avatar_policy import emotion_to_avatar_state
from label_schema import ID_TO_LABEL, LABEL_ZH


@torch.no_grad()
def classify(text: str, model_dir: Path, max_length: int = 128) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu()
    best_id = int(torch.argmax(probs).item())
    emotion = ID_TO_LABEL[best_id]
    confidence = float(probs[best_id].item())
    return {
        "text": text,
        "emotion": emotion,
        "emotion_zh": LABEL_ZH[emotion],
        "confidence": round(confidence, 4),
        "probabilities": {
            ID_TO_LABEL[index]: round(float(value), 4)
            for index, value in enumerate(probs.tolist())
        },
        "avatar": emotion_to_avatar_state(emotion, confidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict emotion for one Chinese text.")
    parser.add_argument("text")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/roberta_wwm_ext"))
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    result = classify(args.text, args.model_dir, args.max_length)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
