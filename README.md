# 面向数字人的流式情绪识别

本项目由基础情绪分类与流式情绪识别两个部分组成：

- 根目录：中文情绪分类基础项目。 基于中文预训练模型微调 SMP2020-EWECT 微博情绪数据，识别 `neutral/happy/angry/sad/fear/surprise` 六类情绪，并输出情绪标签、置信度以及数字人可使用的表情、动作和语气控制字段。

- **`innovation/`：路线 B （见思路.md中的路线B）流式情绪识别项目。** 复用基础分类模型，对持续输入的文本进行滑动窗口推理，并融合历史情绪状态，使得可以**通过token流实时推理情绪**，以便随着文本流实时、平滑地更新数字人情绪表现。


## 1. 流式识别介绍
### 1.1 思路
先用 BERT 对当前窗口预测文本情绪证据：

```text
q_t = BERT(x_{t-k:t})
```

再融合历史情绪记忆：

```text
m_t = weighted_average(p_{t-k:t-1})
p_t = λ q_t + (1 - λ) m_t
```
q_t：即时情绪分布
p_t：平滑后的情绪状态分布
我们选取p_t作为实时识别出的情绪标签，以驱动数字人表情动作等。

### 1.2 演示
演示输入：

```text
我今天本来很开心，但是后来真的有点难过
```

关键窗口的输出如下：

| token index | token | window | emotion | confidence |
| ---: | --- | --- | --- | ---: |
| 7 | 心 | 天本来很开心 | happy | 0.7026 |
| 10 | 是 | 很开心，但是 | happy | 0.8183 |
| 12 | 来 | 心，但是后来 | sad | 0.7080 |
| 14 | 的 | 但是后来真的 | sad | 0.7834 |
| 18 | 过 | 真的有点难过 | sad | 0.8780 |

模型在前半句逐步转向 `happy`；在“但是后来”出现后转向 `sad`，并在“难过”处达到最高悲伤置信度。该结果展示了路线 B 的流式特性：数字人可随着文本推进平滑调整情绪，而不必等整句结束后才切换状态。

## 2. 任务选择

- 数据集：SMP2020-EWECT 微博情绪分类，六类标签为 `neutral/happy/angry/sad/fear/surprise`。
- 主模型：`hfl/chinese-roberta-wwm-ext`。
- 方法：在公开中文情绪数据集上微调预训练模型，并设计流式情绪识别系统（代码文件见`innovation/`目录）。事实上，该思路可用于任意情绪识别模型，将其改造为流式情绪识别系统。
- 应用：输出情绪标签、置信度，情绪标签可用于驱动数字人表情动作等。

## 3. 运行流程

以下命令均在项目根目录执行。运行前请将环境名 `ai_girl` 替换为自己的 Conda 环境名（或直接使用当前 Python 环境）。

### 3.1 下载数据，训练和测试情绪分类模型

```bash
cd /home/fr/FR/AI_Gril/work1/emotion_cls

# 下载并清洗 SMP2020-EWECT 原始微博数据。
# 预期结果：在 data/raw/ 下获得清洗后的原始数据文件。
conda run -n ai_girl python download_data.py

# 将原始数据划分并转换为训练、验证、测试集 CSV。
# 预期结果：在 data/processed/ 下生成 train.csv、dev.csv、test.csv。
conda run -n ai_girl python prepare_data.py

# 微调 hfl/chinese-roberta-wwm-ext 情绪分类模型；batch_size 和 epochs 可按显存调整。
# 预期结果：在 outputs/roberta_wwm_ext/ 下保存模型权重、tokenizer 和训练状态。
conda run -n ai_girl python train.py --batch_size 8 --epochs 3

# 在测试集评估已训练模型。
# 预期结果：终端输出 accuracy、macro-F1 等指标，并在 results/ 下保存分类报告和混淆矩阵。
conda run -n ai_girl python evaluate.py

# 对一条自定义文本进行单句情绪预测。
# 预期结果：终端返回情绪标签、置信度及对应的数字人控制字段。
conda run -n ai_girl python predict.py "我今天真的很开心"

# 运行数字人情绪映射示例。
# 预期结果：展示如何将 emotion 和 confidence 转换为表情、动作、语气和强度。
conda run -n ai_girl python demo_avatar_emotion.py
```

### 3.2 运行路线 B 的流式情绪识别演示：

```bash
# 对输入文本逐 token 执行滑动窗口推理，并融合历史情绪状态。
# --window_size 控制每次推理看到的最近 token 数；
# --lambda_weight 控制即时预测的权重；--memory_decay 控制历史情绪的衰减速度。
# 预期结果：终端以表格逐步输出 token、当前窗口、即时情绪和经平滑后的情绪状态，
#          可直接用于实时驱动数字人的表情、动作和语气。
conda run -n ai_girl python innovation/streaming_emotion.py \
  "我今天本来很开心，但是后来真的有点难过" \
  --model_dir outputs/roberta_wwm_ext \
  --window_size 6 \
  --lambda_weight 0.7 \
  --memory_decay 0.75 \
  --table
```


## 4. 输出文件说明
程序运行后将产生一些文件，说明如下：
- `data/raw/`：SMP2020-EWECT 清洗后的 JSON txt 文件。
- `data/processed/`：转换后的 `train.csv/dev.csv/test.csv`。
- `outputs/roberta_wwm_ext/`：微调后的模型和 tokenizer。
- `results/`：评价指标、分类报告、混淆矩阵。


## 5. 数据来源

- SMP2020-EWECT 官方说明：https://smp2020ewect.github.io/
- 本代码默认从 GitHub 镜像下载清洗版数据：https://github.com/BrownSweater/BERT_SMP2020-EWECT
