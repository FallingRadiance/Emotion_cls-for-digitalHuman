from __future__ import annotations


LABELS = ["neutral", "happy", "angry", "sad", "fear", "surprise"]

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

LABEL_ZH = {
    "neutral": "中性",
    "happy": "开心",
    "angry": "愤怒",
    "sad": "悲伤",
    "fear": "恐惧",
    "surprise": "惊讶",
}


def normalize_label(label: str) -> str:
    value = str(label or "").strip().lower()
    aliases = {
        "无情绪": "neutral",
        "中性": "neutral",
        "积极": "happy",
        "高兴": "happy",
        "开心": "happy",
        "愤怒": "angry",
        "生气": "angry",
        "悲伤": "sad",
        "难过": "sad",
        "恐惧": "fear",
        "害怕": "fear",
        "惊奇": "surprise",
        "惊讶": "surprise",
        "surprised": "surprise",
    }
    value = aliases.get(value, value)
    if value not in LABEL_TO_ID:
        raise ValueError(f"unknown emotion label: {label!r}")
    return value
