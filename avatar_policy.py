from __future__ import annotations

from dataclasses import dataclass

from label_schema import LABEL_ZH


@dataclass(frozen=True)
class AvatarState:
    emotion: str
    emotion_zh: str
    avatar_expression: str
    motion: str
    voice_style: str
    intensity: float


POLICY = {
    "neutral": AvatarState("neutral", LABEL_ZH["neutral"], "neutral", "idle", "calm", 0.20),
    "happy": AvatarState("happy", LABEL_ZH["happy"], "smile", "cheer", "bright", 0.75),
    "angry": AvatarState("angry", LABEL_ZH["angry"], "serious", "restrain", "steady", 0.70),
    "sad": AvatarState("sad", LABEL_ZH["sad"], "sad", "comfort", "soft", 0.65),
    "fear": AvatarState("fear", LABEL_ZH["fear"], "worried", "reassure", "gentle", 0.60),
    "surprise": AvatarState("surprise", LABEL_ZH["surprise"], "surprised", "notice", "lively", 0.68),
}


def emotion_to_avatar_state(emotion: str, confidence: float, threshold: float = 0.55) -> dict[str, object]:
    selected = emotion if confidence >= threshold and emotion in POLICY else "neutral"
    state = POLICY[selected]
    return {
        "emotion": state.emotion,
        "emotion_zh": state.emotion_zh,
        "confidence": round(float(confidence), 4),
        "avatar_expression": state.avatar_expression,
        "motion": state.motion,
        "voice_style": state.voice_style,
        "intensity": state.intensity,
    }
