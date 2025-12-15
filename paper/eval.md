# Evaluations

## 递交

递交结果时用 CLIP ViT-L/14 替换 ViT-B/32

最终测试集 Super Novel 比例约 28 %，Sub Novel 比例约 78 %。

## 数据集划分

- 关于数据集划分，有多种主流方法：
  - 对于 Post-hoc 方法：
    - Train 为纯已知类，Val 为纯已知类，Test 为已知类+未知类
    - Train 为纯已知类，Val 为已知类+未知类，Test 为已知类+未知类（Test 的未知类理论上应和 Val 的未知类不同）
  - 对于 Training-time 方法：
    - Train 为已知类+未知类，Val 为已知类+未知类，Test 为已知类+未知类

目前因为数据集中 superclass 太少，不划分 novel superclass。

## 全局配置

```py
@dataclass
class SplitConfig:
    novel_ratio: float = 0.1  # 每个包含未知类的 split 的未知类比例
    train_ratio: float = 0.8
    val_test_ratio: float = 0.5


@dataclass
class OSRConfig:
    novel_super_index: int = 3
    novel_sub_index: int = 87


@dataclass
class ModelConfig:
    clip_model_id: str = "openai/clip-vit-base-patch32"
    # clip_model_id: str = "openai/clip-vit-large-patch14"  # 升级到 ViT-L/14
    feature_dim: int = 512
    # feature_dim: int = 768  # ViT-L/14 输出 768 维特征


@dataclass
class ExperimentConfig:
    seed: int = 42  # evaluate 时不使用，评估流程中只有 extract features 用到
```

## 基础架构探索

对于阈值设定方法：在正态分布下，Recall = 95% 对应 Z-Score = 1.645，后文都使用该值。

### Baseline: Linear Dual Head + MSP (Temperature = 1) + Quantile

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.Quantile  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1
    prediction_score_temperature: float = 1
```

  [Superclass] Overall     : 91.66% ± 3.27%
  [Superclass] Seen        : 91.66% ± 3.27%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 65.22% ± 0.51%
  [Subclass] Seen          : 87.01% ± 2.13%
  [Subclass] Unseen        : 43.53% ± 2.55%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8540 ± 0.0283

### Baseline: Linear Dual Head + MSP (Temperature = 1) + Z-Score

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1
    prediction_score_temperature: float = 1
```

  [Superclass] Overall     : 98.47% ± 1.07%
  [Superclass] Seen        : 98.47% ± 1.07%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 69.99% ± 2.43%
  [Subclass] Seen          : 88.24% ± 1.58%
  [Subclass] Unseen        : 51.90% ± 5.56%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8540 ± 0.0283

结论：使用 ZScore 设定阈值，是基于分布的统计方法，性能相较于 Baseline 有显著提升。

### Linear Dual Head + MSP + Z-Score

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5
```

  [Superclass] Overall     : 97.25% ± 2.28%
  [Superclass] Seen        : 97.25% ± 2.28%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 72.99% ± 3.39%
  [Subclass] Seen          : 87.33% ± 1.59%
  [Subclass] Unseen        : 58.80% ± 7.47%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8679 ± 0.0265
  
观察：MSP 方法受益于较高温度，T=1.5 时性能最优。高温时，模型关注相对尖锐度

### Linear Dual Head + MSP + Hierarchical Masking + Z-Score

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5
```

  [Superclass] Overall     : 97.25% ± 2.28%
  [Superclass] Seen        : 97.25% ± 2.28%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 72.99% ± 3.39%
  [Subclass] Seen          : 87.33% ± 1.59%
  [Subclass] Unseen        : 58.80% ± 7.47%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8679 ± 0.0265

结论：根据理论，Baseline + Hierarchical Masking >= Baseline 恒成立。

### Gated Dual Head + MSP + Hierarchical Masking + Z-Score

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5
```

  [Superclass] Overall     : 97.74% ± 1.81%
  [Superclass] Seen        : 97.74% ± 1.81%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 73.83% ± 3.88%
  [Subclass] Seen          : 88.03% ± 1.30%
  [Subclass] Unseen        : 59.79% ± 8.48%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8821 ± 0.0234

观察：开启 Feature Gating 后，需要使用更多 epochs 训练。

结论：Gated Dual Head 能使未知 subclass 性能略微提高。

## 具体方法和参数探索

此后若未特别说明，全部使用 Linear Dual Head + Hierarchical Masking + Z-Score

```py
    # 数据划分模式
    test_only_unknown: bool = True  # True: val 不含未知类; False: val含未知类

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645
```

### Baseline: CE + MSP

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5
```

97.74% ± 1.81%, 73.83% ± 3.88%, 88.03% ± 1.30%, 59.79% ± 8.48%, 0.8821 ± 0.0234

### 引出 BCE, MaxSigmoid, Energy

1. 问题：MSP 阈值打分法中用到的 Softmax 的强制归一化导致丢失了幅值信息。方案：使用 Logit-based 的 MaxSigmoid 或 Energy 阈值打分法。

2. 问题：MSP 中使用基于 Softmax 的不保留幅值信息的阈值打分法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是统一的；而使用 Logit-based 的保留幅值信息的阈值打分法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是不统一的。方案：将 Softmax 替换为保留幅值信息的 Sigmoid，并使用 BCE 损失函数。

备注：根据实验测试，不一致的 BCE + MSP, CE + MaxSigmoid, CE + Energy 相比其一致版本的方法，性能均有不同程度下降。

### BCE + MaxSigmoid

```py
    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    prediction_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    validation_score_temperature: float = 0.2
    prediction_score_temperature: float = 0.2
```

  [Superclass] Overall     : 99.61% ± 0.35%
  [Superclass] Seen        : 99.61% ± 0.35%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 75.30% ± 3.70%
  [Subclass] Seen          : 87.29% ± 0.84%
  [Subclass] Unseen        : 63.30% ± 8.03%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8721 ± 0.0365

观察：MaxSigmoid 受益于较低温度，T=0.2 时综合性能最优，但与标准温度结果相差不大，原因未知。温度越低，seen 性能越好，unseen 性能越差。

结论：MaxSigmoid 方法可以使 super 准确率提高到几乎满分。

### BCE + Energy + Quantile

```py
    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.Quantile  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.Energy
    prediction_score_method: OODScoreMethod = OODScoreMethod.Energy
    validation_score_temperature: float = 0.05
    prediction_score_temperature: float = 0.05
```

  [Superclass] Overall     : 91.89% ± 3.53%
  [Superclass] Seen        : 91.89% ± 3.53%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 70.88% ± 1.51%
  [Subclass] Seen          : 87.29% ± 0.89%
  [Subclass] Unseen        : 54.50% ± 3.91%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8755 ± 0.0343

观察：
- Energy 方法由于是无界的，所以 Z-Score 阈值方法不如 Quantile。因此后面讨论 Energy 方法时，都使用 Quantile。
- Energy 方法受益于较低温度，T<=0.05 时性能最优。低温时，模型关注绝对幅值。

## Variant 探索

- CE + MSP (T=1.5):
97.74% ± 1.81%, 73.83% ± 3.88%, 88.03% ± 1.30%, 59.79% ± 8.48%, 0.8821 ± 0.0234
- BCE + MaxSigmoid (T=0.2):
99.61% ± 0.35%, 75.30% ± 3.70%, 87.29% ± 0.84%, 63.30% ± 8.03%, 0.8721 ± 0.0365
- BCE + Energy (T=0.02):
91.89% ± 3.53%, 70.88% ± 1.51%, 87.29% ± 0.89%, 54.50% ± 3.91%, 0.8755 ± 0.0343

变种使用不一致的 Threshold & Prediction 温度 (Tt != Tp)

### CE + MSP

观察：提高 Tt 或降低 Tp：seen 准确率上升，unseen 准确率下降。

平衡 seen & unseen：根据试验，不一致的 Tt & Tp 会严重损害 super 准确率，因此原配置最佳。

### BCE + MaxSigmoid

观察：提高 Tt 或降低 Tp：seen 准确率下降，unseen 准确率上升。

平衡 seen & unseen：无法单纯通过调整 Tt & Tp 来取得平衡，因为这两个参数在这里的边际收益递减非常明显，且Tp < 0.2 会导致 AUROC 显著下降。

### BCE + Energy + Quantile

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

## 总结

super seen, sub overall, sub seen, sub unseen, sub auroc

### 追求一致性：

- CE + MSP (T=1.5):
97.74% ± 1.81%, 73.83% ± 3.88%, 88.03% ± 1.30%, 59.79% ± 8.48%, 0.8821 ± 0.0234
- BCE + MaxSigmoid (T=0.2):
99.61% ± 0.35%, 75.30% ± 3.70%, 87.29% ± 0.84%, 63.30% ± 8.03%, 0.8721 ± 0.0365
- BCE + Energy (T=0.02):
91.89% ± 3.53%, 70.88% ± 1.51%, 87.29% ± 0.89%, 54.50% ± 3.91%, 0.8755 ± 0.0343

### 忽略一致性，追求性能，且尽可能平衡 seen & unseen：

- BCE + Energy (Tt=1.5, Tp=0.02): 
91.87% ± 3.53%, 76.53% ± 2.10%, 82.04% ± 0.85%, 71.03% ± 4.92%, 0.8755 ± 0.0344

### 总结

- 最高 AUROC，应该选 CE + MSP (Tt=Tp=1.5)
- 最高 super seen，选 BCE + MaxSigmoid (Tt=Tp=0.2)

### 关于 Threshold 设定参数

有两种平衡seen和unseen的方法。一种是把 Tt 和 Tp 调成不一样的，另一种是调 Threshold 设定参数。

实验发现：
- 调 Tt Tp：有些情况可以在不掉 super seen 的情况下，把 sub 调匀。
- 调 recall rate 一定会掉 super seen，而且掉的很猛。
- 调 std multiplier 有些情况不掉 super seen，但在有些调 Tt Tp 不掉 super seen 的情况下会掉 super seen。

因此，一定是先把温度调好，再去调 Threshold 设定参数。

## Validation 包含未知类

```py
    # 数据划分模式
    test_only_unknown: bool = False  # True: val 不含未知类; False: val含未知类
```

- CE + MSP (T=1.5):
  98.62% ± 0.50%, 79.57% ± 1.55%, 72.68% ± 2.61%, 84.90% ± 4.12%, 0.8663 ± 0.0119
- BCE + MaxSigmoid (T=0.2):
  99.85% ± 0.12%, 77.58% ± 3.01%, 71.21% ± 3.37%, 82.53% ± 5.16%, 0.8521 ± 0.0350
- BCE + Energy (T=0.02):
  92.44% ± 3.39%, 77.82% ± 2.84%, 68.48% ± 3.37%, 84.87% ± 4.66%, 0.8566 ± 0.0336
- BCE + Energy (Tt=1.5, Tp=0.02): 
  92.36% ± 3.45%, 78.01% ± 3.02%, 65.55% ± 3.14%, 87.33% ± 5.20%, 0.8566 ± 0.0336

## 最终结论

- 最高 AUROC，应该选 CE + MSP (Tt=Tp=1.5)
- 最高 super seen，选 BCE + MaxSigmoid (Tt=Tp=0.2)（理论最完备）

```py
    # 数据划分模式
    test_only_unknown: bool = False  # True: val 不含未知类; False: val含未知类

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.Intersection  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.BCE
    validation_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    prediction_score_method: OODScoreMethod = OODScoreMethod.MaxSigmoid
    validation_score_temperature: float = 0.2
    prediction_score_temperature: float = 0.2
```

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
