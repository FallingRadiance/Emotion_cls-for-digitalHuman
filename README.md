# 面向数字人的中文情绪分类

本目录实现作业 1 的 NLP 任务：基于中文预训练语言模型做微博文本情绪分类，并把分类结果映射为数字人的表情、动作和语气控制信号。

## 任务选择

- 数据集：SMP2020-EWECT 微博情绪分类，六类标签为 `neutral/happy/angry/sad/fear/surprise`。
- 主模型：`hfl/chinese-roberta-wwm-ext`。
- 方法：在公开中文情绪数据集上微调预训练模型，不从零训练。
- 应用：输出情绪标签、置信度和数字人控制字段。

## 运行流程

在项目根目录执行：

```bash
cd /home/fr/FR/AI_Gril/work1/emotion_cls
conda run -n ai_girl python download_data.py
conda run -n ai_girl python prepare_data.py
conda run -n ai_girl python train.py --batch_size 8 --epochs 3
conda run -n ai_girl python evaluate.py
conda run -n ai_girl python predict.py "我今天真的很开心"
conda run -n ai_girl python demo_avatar_emotion.py
```

显存足够时可以把 `--batch_size` 调到 `16`。如果只是测试流程：

```bash
conda run -n ai_girl python train.py --max_train_samples 64 --max_dev_samples 32 --batch_size 8 --epochs 1 --output_dir outputs/smoke
conda run -n ai_girl python evaluate.py --model_dir outputs/smoke --max_samples 32 --output_dir results/smoke
```

## 输出文件

- `data/raw/`：SMP2020-EWECT 清洗后的 JSON txt 文件。
- `data/processed/`：转换后的 `train.csv/dev.csv/test.csv`。
- `outputs/roberta_wwm_ext/`：微调后的模型和 tokenizer。
- `results/`：评价指标、分类报告、混淆矩阵。

## 报告可写重点

1. Transformer 的核心思想：Self-Attention、多头注意力、位置编码、Transformer Encoder。
2. BERT 情绪分类流程：Tokenizer、`[CLS]`、Embedding、Encoder、分类层、Softmax。
3. 微调而非从零训练：利用中文预训练模型的语义表示能力，在情绪标签数据上做任务适配。
4. 数字人应用：将 `emotion + confidence` 转换为 `avatar_expression/motion/voice_style/intensity`。
5. 局限：微博文本口语化、反讽、表情符号、省略语会导致误判；未来可以结合语音、表情和上下文做多模态情绪识别。

## 数据来源

- SMP2020-EWECT 官方说明：https://smp2020ewect.github.io/
- 本代码默认从 GitHub 镜像下载清洗版数据：https://github.com/BrownSweater/BERT_SMP2020-EWECT
