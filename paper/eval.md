# Evaluations

## 基础架构探索

### Baseline: Linear Dual Head + MSP (Temperature = 1)

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 阈值设定
    threshold_method: ThresholdMethod = ThresholdMethod.Quantile  # 阈值设定方法
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1
    prediction_score_temperature: float = 1
```

  [Superclass] Overall     : 95.54% ± 0.46%
  [Superclass] Seen        : 95.54% ± 0.46%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 62.99% ± 1.83%
  [Subclass] Seen          : 89.33% ± 2.39%
  [Subclass] Unseen        : 41.50% ± 3.69%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8423 ± 0.0140

### Linear Dual Head + MSP

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 阈值设定
    threshold_method: ThresholdMethod = ThresholdMethod.Quantile  # 阈值设定方法
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 3.5
```

  [Superclass] Overall     : 95.55% ± 0.55%
  [Superclass] Seen        : 95.55% ± 0.55%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 66.48% ± 3.35%
  [Subclass] Seen          : 88.73% ± 1.73%
  [Subclass] Unseen        : 48.29% ± 6.23%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8667 ± 0.0195
  
观察：MSP 方法受益于较高温度，T=3.5 时性能最优。

### Linear Dual Head + MSP + Hierarchical Masking

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 阈值设定
    threshold_method: ThresholdMethod = ThresholdMethod.Quantile  # 阈值设定方法
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 3.5
```

  [Superclass] Overall     : 95.55% ± 0.55%
  [Superclass] Seen        : 95.55% ± 0.55%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 66.48% ± 3.35%
  [Subclass] Seen          : 88.73% ± 1.73%
  [Subclass] Unseen        : 48.29% ± 6.23%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8667 ± 0.0195

结论：根据理论，Baseline + Hierarchical Masking >= Baseline 恒成立。

### Gated Dual Head + MSP + Hierarchical Masking

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 阈值设定
    threshold_method: ThresholdMethod = ThresholdMethod.Quantile  # 阈值设定方法
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 3.5
```

  [Superclass] Overall     : 95.39% ± 0.38%
  [Superclass] Seen        : 95.39% ± 0.38%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 68.59% ± 1.21%
  [Subclass] Seen          : 89.01% ± 2.32%
  [Subclass] Unseen        : 51.90% ± 3.52%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8808 ± 0.0171

观察：开启 Feature Gating 后，需要使用更多 epochs 训练。

结论：Gated Dual Head 能使未知subclass准确率显著提高。

## 具体方法和参数探索

此后全部使用 Linear Dual Head + Hierarchical Masking

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 阈值设定
    threshold_method: ThresholdMethod = ThresholdMethod.Quantile  # 阈值设定方法
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关
```

### Baseline: CE + MSP

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 3.5
```

  [Superclass] Overall     : 95.39% ± 0.38%
  [Superclass] Seen        : 95.39% ± 0.38%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 68.59% ± 1.21%
  [Subclass] Seen          : 89.01% ± 2.32%
  [Subclass] Unseen        : 51.90% ± 3.52%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8808 ± 0.0171

### CE + Energy

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.Energy
    prediction_score_method: OODScoreMethod = OODScoreMethod.Energy
    validation_score_temperature: float = 0.02
    prediction_score_temperature: float = 0.02
```

观察：Energy 方法受益于较低温度，T=0.02 时性能最优。

  [Superclass] Overall     : 95.43% ± 0.62%
  [Superclass] Seen        : 95.43% ± 0.62%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 64.08% ± 1.29%
  [Subclass] Seen          : 88.22% ± 2.65%
  [Subclass] Unseen        : 44.38% ± 4.09%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8590 ± 0.0191

### 阶段性结论

1. MSP 受益于较高温度（T=3.5 时性能最优）：高温时，模型关注相对尖锐度。
2. Energy 方法受益于较低温度（T=0.02 时性能最优）：低温时，模型关注绝对幅值。
3. 问题：Softmax 的强制归一化导致丢失了幅值信息；方案：使用基于 Logits 的 Energy 方法。
4. MSP 中使用基于 Softmax 的不保留幅值信息的阈值打分方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是统一的；而 Energy 方法 中使用基于 Logits 的保留幅值信息的阈值打分方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是不统一的。所以理论上应将 Softmax 替换为保留幅值信息的 Sigmoid，并使用 BCE 损失函数。

### BCE + Energy

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.Energy
    prediction_score_method: OODScoreMethod = OODScoreMethod.Energy
    validation_score_temperature: float = 0.02
    prediction_score_temperature: float = 0.02
```

  [Superclass] Overall     : 95.34% ± 0.33%
  [Superclass] Seen        : 95.34% ± 0.33%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 68.01% ± 2.82%
  [Subclass] Seen          : 89.01% ± 1.82%
  [Subclass] Unseen        : 50.86% ± 4.91%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8593 ± 0.0251

结论：对 Energy 配置，将损失函数从 Softmax + CE 替换为 Sigmoid + BCE，性能有显著提升。

### BCE + MaxSigmoid

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    prediction_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    validation_score_temperature: float = 1
    prediction_score_temperature: float = 1
```

  [Superclass] Overall     : 95.34% ± 0.33%
  [Superclass] Seen        : 95.34% ± 0.33%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 68.01% ± 2.82%
  [Subclass] Seen          : 89.01% ± 1.82%
  [Subclass] Unseen        : 50.86% ± 4.91%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8593 ± 0.0251

观察：只要 T >= 1，结果将完全保持一致。

结论：对于 Sigmoid + BCE 损失函数，使用 MaxSigmoid 或 Energy 阈值打分方法，性能相同。

## Variant 探索

- CE + MSP (T=3.5):
95.39% ± 0.38%, 68.59% ± 1.21%, 89.01% ± 2.32%, 51.90% ± 3.52%, 0.8808 ± 0.0171
- BCE + Energy (T=0.02):
95.34% ± 0.33%, 68.01% ± 2.82%, 89.01% ± 1.82%, 50.86% ± 4.91%, 0.8593 ± 0.0251
- BCE + MaxSigmoid (T=1):
95.34% ± 0.33%, 68.01% ± 2.82%, 89.01% ± 1.82%, 50.86% ± 4.91%, 0.8593 ± 0.0251

变种可以为：

- BCE + MSP
- 使用不一致的 Threshold & Prediction 温度 (Tt != Tp)

### BCE + MSP

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 3.5
```

  [Superclass] Overall     : 95.30% ± 0.42%
  [Superclass] Seen        : 95.30% ± 0.42%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 66.33% ± 3.06%
  [Subclass] Seen          : 89.20% ± 1.86%
  [Subclass] Unseen        : 47.65% ± 5.78%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8577 ± 0.0238

结论：与理论相符，不一致的训练与推理方法导致性能下降。

### CE + MSP

对于 CE + MSP，经过测试，若要维持 88% + 的 seen 分类准确率，原配置 T=3.5 最佳。

观察：提高 Tt 或降低 Tp：seen 准确率上升，unseen 准确率下降。

平衡 seen & unseen：根据试验，不一致的 Tt & Tp 会严重损害 super 准确率，因此无法达成平衡。

### BCE + Energy

对于 BCE + Energy，经过测试，若要维持 88% + 的 seen 分类准确率，原配置 T=0.02 最佳。

观察：提高 Tt 或降低 Tp：seen 准确率下降，unseen 准确率上升。

平衡 seen & unseen：最佳配置为 Tt = 2, Tp = 0.02，提升幅度为 7%。

  [Superclass] Overall     : 95.34% ± 0.33%
  [Superclass] Seen        : 95.34% ± 0.33%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 75.79% ± 2.42%
  [Subclass] Seen          : 79.59% ± 1.99%
  [Subclass] Unseen        : 72.70% ± 5.18%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8593 ± 0.0251

### BCE + MaxSigmoid

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    prediction_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    validation_score_temperature: float = 3.5
    prediction_score_temperature: float = 1
```

  [Superclass] Overall     : 99.47% ± 0.28%
  [Superclass] Seen        : 99.47% ± 0.28%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 70.27% ± 3.14%
  [Subclass] Seen          : 89.44% ± 0.86%
  [Subclass] Unseen        : 54.63% ± 5.99%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8593 ± 0.0251

结论：对于 BCE + 双 MaxSigmoid 方法，Tt = 3.5, Tp = 1，全性能均有显著提升，尤其是 super 性能取得了目前的最佳。这是一个很奇怪的事情，因为根据前面的试验，取不一样的 Tt & Tp，是在 seen 与 unseen 之间取得平衡，而这里直接提升了全性能。

观察：提高 Tt 或降低 Tp：seen 准确率下降，unseen 准确率上升。

平衡 seen & unseen：最佳配置为 Tt = 10, Tp = 0.5，提升幅度为 1%。这里似乎无法单纯通过调整 Tt & Tp 来取得平衡，因为这两个参数在这里的边际收益递减非常明显，且Tp <= 0.2 会导致 AUROC 显著下降。

  [Superclass] Overall     : 99.81% ± 0.17%
  [Superclass] Seen        : 99.81% ± 0.17%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 71.14% ± 3.48%
  [Subclass] Seen          : 88.89% ± 0.92%
  [Subclass] Unseen        : 56.68% ± 6.66%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8593 ± 0.0251

## 总结

super seen, sub overall, sub seen, sub unseen, sub auroc

### 追求一致性：

- CE + MSP (T=3.5):
95.39% ± 0.38%, 68.59% ± 1.21%, 89.01% ± 2.32%, 51.90% ± 3.52%, 0.8808 ± 0.0171
- BCE + Energy (T=0.02):
95.34% ± 0.33%, 68.01% ± 2.82%, 89.01% ± 1.82%, 50.86% ± 4.91%, 0.8593 ± 0.0251
- BCE + MaxSigmoid (T=1):
95.34% ± 0.33%, 68.01% ± 2.82%, 89.01% ± 1.82%, 50.86% ± 4.91%, 0.8593 ± 0.0251

- 应该选 CE + MSP (T=3.5)，因为其 AUROC 显著更高。

### 忽略一致性，追求性能，且尽可能平衡 seen & unseen：

- CE + MSP (Tt=Tp=3.5):      
95.07% ± 0.12%, 71.57% ± 1.76%, 87.65% ± 0.97%, 57.86% ± 3.20%, 0.8940 ± 0.0077
- BCE + Energy (Tt=2, Tp=0.02): 
95.34% ± 0.21%, 75.79% ± 2.42%，79.59% ± 1.99%, 72.70% ± 5.18%, 0.8593 ± 0.0251
- BCE + MaxSigmoid (Tt=10, Tp=0.5):
99.81% ± 0.17%, 71.14% ± 3.48%, 88.89% ± 0.92%, 56.68% ± 6.66%, 0.8593 ± 0.0251

- 最高 AUROC，应该选 CE + MSP (Tt=Tp=3.5)
- 最高 sub overall，选 BCE + Energy & MaxSigmoid (Tt=0.02, Tp=2)
- 最高 super seen，选 BCE + MaxSigmoid (Tt=10, Tp=0.5)

### 关于 Recall Rate 设定

有两种平衡seen和unseen的方法。一种是把 Tt 和 Tp 调成不一样的，另一种是调 recall rate。

实验发现 Tt Tp 调了以后，有些时候可以在不掉 super seen 的情况下，把 sub 调匀。但是调 recall rate 一定会掉 super seen，而且掉的很猛。

所以一定是先把温度调好，再去调 recall rate。

# CAC 
## v1.0

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 400
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    alpha: float = 10
    lambda_w: float
    anchor_mode: "axis_aligned"
```
[Subclass] Overall : 68.63% ± 0.64% 
[Subclass] Seen : 94.86% ± 0.29% 
[Subclass] Unseen : 46.25% ± 1.31% 
[Subclass] AUROC : 0.8642 ± 0.0003

|   lambda_w  | Subclass Overall |  Subclass Seen  | Subclass Unseen| Subclass AUROC | 
|-------------|------------------|-----------------|----------------|----------------|
| 0.1         | 68.63% ± 0.64%   | 94.86% ± 0.29%  | 46.25% ± 1.31% | 0.8642 ± 0.0003|
| 0           | 72.00% ± 0.47%   | 95.10% ± 0.12%  | 52.31% ± 0.87% | 0.8908 ± 0.0010|

在模型放弃最小化样本与其真实类别锚点之间的距离（lambda_w=0）时，已知和未知subclass准确率均提高，可能是因为即使样本较锚点远，模型已经能够划分开已知和未知
  
## v1.1
```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 300
    target_recall: float = 0.95
    # 模型参数
    alpha: float = 8.0
    lambda_w: float
    anchor_mode: "uniform_hypersphere"
```

|   lambda_w  | Subclass Overall |  Subclass Seen | Subclass Unseen| Subclass AUROC | 
|-------------|------------------|----------------|----------------|----------------|
| 0.1         | 69.66% ± 1.09%   | 93.80% ± 0.27% | 49.06% ± 1.99% | 0.8715 ± 0.0012|
| 0           | 71.70% ± 0.44%   | 94.71% ± 0.35% | 52.07% ± 0.57% | 0.8859 ± 0.0005|

无提升

## v1.2
```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 300
    target_recall: float = 0.95
    # 模型参数
    alpha: float = 8.0
    lambda_w: float = 0
    se_reduction: float = 4
    anchor_mode: "axis_aligned"
```
| se_reduction | Subclass Overall |  Subclass Seen  | Subclass Unseen | Subclass AUROC | 
|--------------|------------------|-----------------|-----------------|----------------|
| 2            | 71.95% ± 0.87%   | 94.82% ± 0.24%  | 52.44% ± 1.59%  | 0.8930 ± 0.0013|
| 4            | 72.53% ± 0.97%   | 94.55% ± 0.26%  | 53.75% ± 1.86%  | 0.8913 ± 0.0030|
| 8            | 71.52% ± 0.76%   | 94.78% ± 0.16%  | 51.67% ± 1.46%  | 0.8925 ± 0.0018|

se_reduction=4相较于v1.0有更好表现

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 300
    target_recall: float = 0.95
    # 模型参数
    alpha: float = 8.0
    lambda_w: float = 0
    se_reduction: float
    anchor_mode: "uniform_hypersphere"
```

| se_reduction | Subclass Overall  | Subclass Seen   | Subclass Unseen | Subclass AUROC | 
|--------------|-------------------|-----------------|-----------------|----------------|
| 2            | 71.71% ± 0.66%    | 94.59% ± 0.29%  | 52.21% ± 1.30%  | 0.8857 ± 0.0027|
| 4            | 71.84% ± 0.69%    | 94.47% ± 0.19%  | 52.54% ± 1.21%  | 0.8864 ± 0.0011|
| 8            | 72.08% ± 0.61%    | 94.63% ± 0.53%  | 52.84% ± 1.03%  | 0.8877 ± 0.0013|

# OpenMax
## v1.0 LinearClassifier + OpenMax
```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    # 模型参数
    alpha: int
    weibull_tail_size: int
    distance_type = "cosine"
```

| alpha | weibull_tail_size | Subclass Overall | Subclass Seen   | Subclass Unseen  | Subclass AUROC   | 
|-------|-------------------|------------------|-----------------|------------------|------------------|
| 1     | 3                 | 71.55% ± 0.14%   | 92.63% ± 0.29%  | 53.58% ± 0.25%   | 0.9074 ± 0.0009  |
| 2     | 3                 | 77.13% ± 0.25%   | 91.84% ± 0.24%  | 64.58% ± 0.45%   | 0.9293 ± 0.0011  |
| 3     | 3                 | 80.18% ± 0.23%   | 90.86% ± 0.24%  | 71.07% ± 0.33%   | 0.9378 ± 0.0011  |
| 3     | 4                 | 81.57% ± 0.11%   | 89.29% ± 0.32%  | 74.98% ± 0.13%   | 0.9345 ± 0.0012  |
| 3     | 5                 | 82.76% ± 0.24%   | 88.00% ± 0.15%  | 78.29% ± 0.34%   | 0.9310 ± 0.0014  |
| 3     | 10                | 84.64% ± 0.17%   | 81.18% ± 0.18%  | 87.59% ± 0.16%   | 0.9188 ± 0.0010  |

## v1.1 CAC + OpenMax
``` py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 300
    # CAC 参数
    alpha_CAC: float
    lambda_w: float
    # OpenMax 参数
    alpha: int
    weibull_tail_size: int
    distance_type = "cosine"
```
| alpha_CAC | lambda_w | se_reduction | alpha_OpenMax | weibull_tail_size | Subclass Overall   | Subclass Seen   | Subclass Unseen | Subclass AUROC  | 
|-----------|----------|--------------|---------------|-------------------|--------------------|-----------------|-----------------|-----------------|
| 8         | 0        | 4            | 3             | 10                | 84.37% ± 0.44%     | 74.86% ± 0.66%  | 92.47% ± 0.42%  | 0.9053 ± 0.0031 |
| 10        | 0        | 4            | 3             | 10                | 85.05% ± 0.27%     | 77.06% ± 0.53%  | 91.87% ± 0.57%  | 0.9121 ± 0.0030 |
| 10        | 0        | 4            | 3             | 5                 | 85.96% ± 0.66%     | 84.55% ± 0.50%  | 87.16% ± 0.96%  | 0.9312 ± 0.0031 |
| 10        | 0        | 4            | 3             | 3                 | 85.00% ± 0.69%     | 88.35% ± 0.59%  | 82.14% ± 1.01%  | 0.9377 ± 0.0027 |
| 10        | 0        | -1           | 3             | 3                 | 85.67% ± 0.07%     | 89.65% ± 0.15%  | 82.27% ± 0.15%  | 0.9439 ± 0.0008 |
| 10        | 0        | -1           | 3             | 5                 | 86.23% ± 0.07%     | 85.18% ± 0.36%  | 87.12% ± 0.24%  | 0.9367 ± 0.0007 |
| 10        | 0.1      | -1           | 3             | 5                 | **86.62% ± 0.16%** | 84.47% ± 0.38%  | 88.46% ± 0.11%  | 0.9281 ± 0.0026 |
| 10        | 0.1      | -1           | 5             | 5                 | 84.87% ± 0.12%     | 76.51% ± 0.29%  | 92.01% ± 0.07%  | 0.9273 ± 0.0024 |
| 8         | 0.1      | -1           | 3             | 5                 | 86.14% ± 0.18%     | 82.59% ± 0.29%  | 89.16% ± 0.20%  | 0.9208 ± 0.0016 |
| 10        | 0.2      | -1           | 3             | 5                 | 86.10% ± 0.20%     | 82.75% ± 0.28%  | 88.96% ± 0.18%  | 0.9189 ± 0.0030 |
| 10        | 0.1      | 4            | 3             | 5                 | 86.10% ± 0.23%     | 83.45% ± 0.36%  | 88.36% ± 0.17%  | 0.9234 ± 0.0042 |
