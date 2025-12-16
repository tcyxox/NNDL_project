# Evaluations

## 递交

递交结果时用 CLIP ViT-L/14 替换 ViT-B/32

最终测试集 Super Novel 比例约 28 %，Sub Novel 比例约 78 %。

## 数据集划分

- 关于数据集划分，有多种主流方法：
  - 对于 Post-hoc 方法：
    - Strict: Train 为纯已知类，Val 为纯已知类，Test 为已知类+未知类
    - Guided: Train 为纯已知类，Val 为已知类+未知类，Test 为已知类+未知类（Test 的未知类理论上应和 Val 的未知类不同）
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

## 基础架构探索（Ablation Study 消融实验）

对于阈值设定方法：在正态分布下，Recall = 95% 对应 Z-Score = 1.645，后文都使用该值。

### Baseline: Linear Dual Head + MSP (Temperature = 1) + Quantile

```py
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50

    # 数据划分模式
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.Quantile  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 模型选择
    enable_hierarchical_masking: bool = False  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = False  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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

此后若未特别说明，全部使用 Linear Dual Head + Hierarchical Masking + Z-Score，即以下配置：

```py
    # 数据划分模式
    val_include_novel: bool = False  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
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

## 总结

super seen, sub overall, sub seen, sub unseen, sub auroc

- CE + MSP (T=1.5):
97.74% ± 1.81%, 73.83% ± 3.88%, 88.03% ± 1.30%, 59.79% ± 8.48%, 0.8821 ± 0.0234
- BCE + MaxSigmoid (T=0.2):
99.61% ± 0.35%, 75.30% ± 3.70%, 87.29% ± 0.84%, 63.30% ± 8.03%, 0.8721 ± 0.0365
- BCE + Energy (T=0.02):
91.89% ± 3.53%, 70.88% ± 1.51%, 87.29% ± 0.89%, 54.50% ± 3.91%, 0.8755 ± 0.0343

## Validation 包含未知类

此时，会自动选择 EER 方法。

```py
    # 数据划分模式
    val_include_novel: bool = True  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel
```

- CE + MSP (T=1.5):
  98.62% ± 0.50%, 79.57% ± 1.55%, 72.68% ± 2.61%, 84.90% ± 4.12%, 0.8663 ± 0.0119
- BCE + MaxSigmoid (T=0.2):
  99.85% ± 0.12%, 77.58% ± 3.01%, 71.21% ± 3.37%, 82.53% ± 5.16%, 0.8521 ± 0.0350
- BCE + Energy (T=0.02):
  92.44% ± 3.39%, 77.82% ± 2.84%, 68.48% ± 3.37%, 84.87% ± 4.66%, 0.8566 ± 0.0336
- BCE + Energy (Tt=1.5, Tp=0.02): 
  92.36% ± 3.45%, 78.01% ± 3.02%, 65.55% ± 3.14%, 87.33% ± 5.20%, 0.8566 ± 0.0336

## 问题

根据在professor测试集上的测试，MaxSigmoid 会对 super seen 的判定特别自信，在 validation 中加入 novel superclass 可以缓解，但问题依然存在；而 MSP 则基本没有该问题。

## 标准 Calibration 流程

以上是快速验证，为了得到最好的方法。以下对最好的 CE + MSP方法，使用标准校准流程进行准确评估。

流程如下：

术语：full train完整训练集，train剔除test的训练集，sub train剔除test再剔除val的训练集

1. 输入full train, test_sub_novel_ratio=10%, test_ratio=20%。在full train上划出10%的subclass作为novel交给test，然后在剩余的90%里部分划20%-10%=10%给test。现在test有了基本平衡数量的known和sub novel，并且总占比约为20%。输出train, test。
2. 输入train, test, val_sub_novel_ratio=10%, val_ratio=20%, val_include_nodel=true, force_super_novel=true/false，seeds=[...]，进入阈值计算循环
  1. 如果force_super_novel为true，随机选择一个superclass设为novel交给val。计算sub novel距离10%缺多少比例，补足该比例交给val。如果此时sub known不足sub novel的50%，则补足交给val。然后剩余的部分划20%-10%=10%给val。现在val有了基本平衡数量的known和novel，并且总占比约为20%（如果这个novel super没有占过超过10%）。输出subtrain, val。（如果val_include_novel为false，没有其它步骤，直接划20%给val）
  2. 在subtrain上训练
  3. 用subtrain在val上推理并计算阈值
3. 取平均阈值
4. 在train上训练
5. 在test上推理并得到统计

1-5按照seed=[]进行外轮循环，得到最终报告。

## Super Novel 引入 Val 测试

```py
    # 数据划分模式
    val_include_novel: bool = True  # Val 中是否含未知类
    force_super_novel: bool = False  # 是否在 Val 中强制引入 Super Novel
```

经过测试，如果force_super_novel为true，会导致sub novel占比过大，使得阈值设定出现问题，sub unseen准确率急剧下降。因此force_super_novel为false。

## Calibration 流程对比（用新流程跑，所以不会复现前面的结果）

使用 CE + MSP (T1.5) + ViT-B/32 + epochs=100

- val_include_novel=False
  98.84% ± 0.63%, 71.98% ± 3.23%, 90.21% ± 0.66%, 50.34% ± 6.97%, 0.8799 ± 0.0276
- val_include_novel=True
  98.98% ± 0.81%, 79.05% ± 3.59%, 81.52% ± 2.53%, 76.00% ± 7.42%, 0.8799 ± 0.0276
- ViT-L/14, epochs=75, val_include_novel=True
  97.07% ± 3.05%, 83.02% ± 3.69%, 84.84% ± 2.84%, 80.74% ± 8.65%, 0.9117 ± 0.0318

## 最终结论

选 CE + MSP (T1.5)

```py
    # 数据划分模式
    val_include_novel: bool = False  # True: val 不含未知类; False: val含未知类

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5
```
