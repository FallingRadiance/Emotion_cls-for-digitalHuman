# 创新任务：路线 B 流式情绪状态估计

本目录实现路线 B：不额外训练 token 级模型，而是复用基础任务训练出的句子/窗口级情绪分类模型，做滑动窗口推理和历史情绪记忆融合。

## 公式

```text
r_t = BERT(x_{t-k:t})
m_t = weighted_average(s_{t-k:t-1})
s_t = lambda * r_t + (1 - lambda) * m_t
```

- `r_t`：当前窗口即时情绪预测。
- `m_t`：最近一段历史情绪状态的加权记忆。
- `s_t`：最终用于驱动数字人的平滑情绪状态。

## 模型配置

先完成基础任务训练，得到模型目录：

```bash
cd /home/fr/FR/AI_Gril/work1/emotion_cls
conda run -n ai_girl python train.py --batch_size 8 --epochs 3
```

默认会使用：

```text
outputs/roberta_wwm_ext
```

如果模型目录不同，修改 `innovation/streaming_config.py` 里的 `MODEL_DIR`，或运行时传 `--model_dir`。

## 运行

```bash
cd /home/fr/FR/AI_Gril/work1/emotion_cls
conda run -n ai_girl python innovation/streaming_emotion.py \
  "我今天本来很开心，但是后来真的有点难过" \
  --model_dir outputs/roberta_wwm_ext \
  --window_size 6 \
  --lambda_weight 0.7 \
  --memory_decay 0.75 \
  --table
```

完整 JSON 输出：

```bash
conda run -n ai_girl python innovation/streaming_emotion.py \
  "我今天本来很开心，但是后来真的有点难过" \
  --model_dir outputs/roberta_wwm_ext \
  --output results/streaming_demo.json
```

每个 token 会输出当前窗口、即时预测 `r_t`、历史记忆 `m_t`、平滑状态 `s_t` 和数字人控制字段。
