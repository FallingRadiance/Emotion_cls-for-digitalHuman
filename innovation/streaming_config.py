from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 使用前先训练基础情绪分类模型：
# conda run -n ai_girl python train.py --batch_size 8 --epochs 3
# 训练完成后默认模型目录就是 PROJECT_ROOT / "outputs" / "roberta_wwm_ext"。
# 如果你把模型保存到了其他目录，改这里，或运行脚本时传 --model_dir。
MODEL_DIR = PROJECT_ROOT / "outputs" / "roberta_wwm_ext"

# 当前 token 向前看的窗口大小。中文场景先按字符流处理，6 表示最多看最近 6 个字符。
WINDOW_SIZE = 6

# 融合权重 lambda：越大越相信当前窗口即时预测 r_t，越小越相信历史情绪记忆 m_t。
LAMBDA_WEIGHT = 0.70

# 历史记忆衰减系数：越接近 1，越保留更久的历史；越小，越强调最近状态。
MEMORY_DECAY = 0.75

# 数字人动作触发阈值：最终状态最大概率低于该值时，avatar_policy 会回退到 neutral。
CONFIDENCE_THRESHOLD = 0.55
