# 报告结构建议

## 1 引言

说明数字人在交互中需要根据用户文本实时感知情绪，情绪分类模块可以辅助表情、动作和语气变化。

## 2 Transformer 与 BERT 原理

介绍 Self-Attention、Multi-Head Attention、Position Embedding、Transformer Encoder，以及 BERT 如何使用 `[CLS]` 向量完成分类。

## 3 数据集

使用 SMP2020-EWECT 微博情绪分类数据集。数据包含通用微博和疫情微博两个领域，标签为 neutral、happy、angry、sad、fear、surprise。

## 4 方法

使用 `hfl/chinese-roberta-wwm-ext` 作为预训练模型，在情绪分类数据上微调。输入文本经 tokenizer 转为 token ids 和 attention mask，送入 BERT/RoBERTa 编码后取 `[CLS]` 表示，接线性分类层输出六类概率。

## 5 实验

写明训练参数：max length 128、batch size 8 或 16、epoch 3、learning rate 2e-5、macro-F1 作为主要指标。

## 6 结果与分析

展示 accuracy、macro-F1、分类报告和混淆矩阵。分析容易混淆的类别，例如 fear 与 sad、happy 与 neutral。

## 7 数字人应用设计

将情绪分类结果映射为数字人控制信号：

| 情绪 | 表情 | 动作 | 语气 |
| --- | --- | --- | --- |
| neutral | neutral | idle | calm |
| happy | smile | cheer | bright |
| angry | serious | restrain | steady |
| sad | sad | comfort | soft |
| fear | worried | reassure | gentle |
| surprise | surprised | notice | lively |

低置信度时回退到 neutral，并使用平滑机制避免数字人频繁跳变。

## 8 总结

总结本文完成了中文文本情绪分类微调，并将其扩展为数字人实时情绪辅助模块。未来可以加入语音情绪、面部表情和多轮上下文。
