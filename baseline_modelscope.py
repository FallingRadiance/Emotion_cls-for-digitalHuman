from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an off-the-shelf ModelScope Chinese emotion classifier baseline.")
    parser.add_argument("text", nargs="?", default="我今天真的很难过，感觉什么都做不好。")
    parser.add_argument("--model_id", default="iic/nlp_structbert_emotion-classification_chinese-base")
    parser.add_argument("--model_revision", default="v1.0.0")
    args = parser.parse_args()

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    classifier = pipeline(
        Tasks.text_classification,
        args.model_id,
        model_revision=args.model_revision,
    )
    start = time.perf_counter()
    result = classifier(input=args.text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(json.dumps({"text": args.text, "elapsed_ms": elapsed_ms, "raw_result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
