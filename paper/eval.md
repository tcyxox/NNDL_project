# Evaluations

## Baseline: 独立双头 + MSP

### 参数配置

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = False  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = False  # 训练时使用 SE Feature Gating
    enable_energy: bool = False  # 使用 Energy-based OOD 检测 替代 MSP
```

### 评估结果

  [Superclass] Overall     : 95.40% ± 0.16%
  [Superclass] Seen        : 95.40% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 62.17% ± 0.63%
  [Subclass] Seen          : 88.82% ± 0.21%
  [Subclass] Unseen        : 39.43% ± 1.16%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8539 ± 0.0016

## 独立双头 + MSP + Hierarchical Masking

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = False  # 训练时使用 SE Feature Gating
    enable_energy: bool = False  # 使用 Energy-based OOD 检测 替代 MSP
```

### 评估结果

  [Superclass] Overall     : 95.40% ± 0.16%
  [Superclass] Seen        : 95.40% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 62.17% ± 0.63%
  [Subclass] Seen          : 88.82% ± 0.21%
  [Subclass] Unseen        : 39.43% ± 1.16%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8539 ± 0.0016

结论：根据理论，Baseline + Hierarchical Masking >= Baseline（实验结果为取等情况）。

## Feature Gating 联合双头 + MSP + Hierarchical Masking

### 参数配置

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = True  # 训练时使用 SE Feature Gating
    enable_energy: bool = False  # 使用 Energy-based OOD 检测 替代 MSP
```

### Temperature 调优

| Temperature | Subclass Overall   | Subclass Unseen    | Subclass AUROC     |
|-------------|--------------------|--------------------|--------------------|
| 0.8         | 64.39% ± 2.40%     | 44.90% ± 4.50%     | 0.8587 ± 0.0049    |
| 1.0         | 65.90% ± 2.39%     | 47.29% ± 4.50%     | 0.8649 ± 0.0053    |
| 1.2         | 66.84% ± 2.71%     | 49.13% ± 5.14%     | 0.8710 ± 0.0055    |
| 1.5         | 67.85% ± 3.19%     | 51.10% ± 6.31%     | 0.8786 ± 0.0061    |
| 2.0         | 69.22% ± 2.72%     | 53.65% ± 5.14%     | 0.8859 ± 0.0070    |
| **2.5**     | **69.64% ± 2.28%** | **54.62% ± 4.40%** | **0.8879 ± 0.0078**|
| 3.0         | 69.83% ± 2.05%     | 54.91% ± 4.02%     | 0.8876 ± 0.0083    |

观察：MSP 方法受益于较高温度，T=2.5 时性能最优。

### 评估结果

  [Superclass] Overall     : 95.16% ± 0.27%
  [Superclass] Seen        : 95.16% ± 0.27%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 69.64% ± 2.2%
  [Subclass] Seen          : 87.25% ± 0.33%
  [Subclass] Unseen        : 54.62% ± 4.40%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8879 ± 0.0078

结论：Feature Gating 联合双头 能使未知subclass准确率显著提高。

## Feature Gating 联合双头 + Energy-based OOD + Hierarchical Masking

### 参数配置

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = True  # 训练时使用 SE Feature Gating
    enable_energy: bool = True  # 使用 Energy-based OOD 检测 替代 MSP
```

### Temperature Scaling 实验

| Temperature | Subclass Overall   | Subclass Unseen    | Subclass AUROC     |
|-------------|--------------------|--------------------|--------------------|
| 0.1         | 65.57% ± 0.00%     | 48.01% ± 0.00%     | 0.8692 ± 0.0000    |
| **0.15**    | **65.90% ± 1.33%** | **47.89% ± 2.82%** | **0.8726 ± 0.0091**|
| 0.2         | 65.85% ± 1.30%     | 47.76% ± 2.77%     | 0.8725 ± 0.0091    |
| 0.3         | 65.81% ± 1.28%     | 47.63% ± 2.71%     | 0.8724 ± 0.0090    |
| 0.4         | 65.61% ± 1.41%     | 47.22% ± 2.90%     | 0.8722 ± 0.0089    |
| 0.5         | 65.25% ± 1.52%     | 46.56% ± 3.08%     | 0.8718 ± 0.0089    |
| 0.6         | 65.04% ± 1.77%     | 46.19% ± 3.52%     | 0.8714 ± 0.0088    |
| 0.8         | 64.84% ± 1.84%     | 45.95% ± 3.69%     | 0.8700 ± 0.0088    |
| 1.0         | 64.31% ± 2.08%     | 45.22% ± 4.21%     | 0.8678 ± 0.0089    |
| 1.2         | 63.65% ± 2.26%     | 43.98% ± 4.50%     | 0.8647 ± 0.0091    |

观察：Energy-based 方法受益于较低温度，T=0.15 时性能最优。

### 评估结果

  [Superclass] Overall     : 95.22% ± 0.20%
  [Superclass] Seen        : 95.22% ± 0.20%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 65.90% ± 1.33%
  [Subclass] Seen          : 87.02% ± 0.90%
  [Subclass] Unseen        : 47.89% ± 2.82%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8726 ± 0.0091

### 阶段性结论

1. Feature Gating 联合双头 + Hierarchical Masking 是一定要使用的。
2. MSP 受益于较高温度（T=2.5 时性能最优）：高温时，模型关注相对尖锐度。
3. Energy-based OOD 受益于较低温度（T=0.15 时性能最优）：低温时，模型关注绝对幅值。
4. 问题：Softmax 的强制归一化导致丢失了幅值信息；方案：使用基于 Logits 的 Energy-based OOD。
5. MSP 中使用基于 Softmax 的不保留幅值信息的阈值方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是统一的；而 Energy-based OOD 中使用基于 Logits 的保留幅值信息的阈值方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，十部统一的。所以应将 Softmax 替换为保留幅值信息的 Sigmoid。

## CAC v1.0

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    epochs: int = 400
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    alpha: float = 10
    lambda_w: float = 0.1
```

  [Subclass] Overall            : 68.63% ± 0.64%
  [Subclass] Seen               : 94.86% ± 0.29%
  [Subclass] Unseen             : 46.25% ± 1.31%
  [Subclass] AUROC              : 0.8642 ± 0.0003
