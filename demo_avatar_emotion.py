from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict import classify


def smooth(previous: dict[str, float], current: dict[str, float], alpha: float) -> dict[str, float]:
    keys = set(previous) | set(current)
    return {
        key: round(alpha * previous.get(key, 0.0) + (1.0 - alpha) * current.get(key, 0.0), 4)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show how text emotion predictions can drive avatar states.")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/roberta_wwm_ext"))
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight of previous emotion state.")
    parser.add_argument(
        "texts",
        nargs="*",
        default=[
            "今天终于把任务完成了，太开心了。",
            "可是我还是有点担心明天会不会出问题。",
            "如果真的失败了我会很难过。",
        ],
    )
    args = parser.parse_args()

    state: dict[str, float] = {}
    outputs: list[dict[str, object]] = []
    for text in args.texts:
        result = classify(text, args.model_dir)
        state = smooth(state, result["probabilities"], args.alpha)
        outputs.append(
            {
                "text": text,
                "prediction": {
                    "emotion": result["emotion"],
                    "emotion_zh": result["emotion_zh"],
                    "confidence": result["confidence"],
                },
                "smoothed_distribution": state,
                "avatar": result["avatar"],
            }
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
