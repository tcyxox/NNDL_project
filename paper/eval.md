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
| 0.8         | 64.80% ± 2.70%     | 45.15% ± 4.96%     | 0.8587 ± 0.0049    |
| 1.0         | 65.90% ± 2.39%     | 47.29% ± 4.50%     | 0.8649 ± 0.0053    |
| 1.2         | 66.84% ± 2.71%     | 49.13% ± 5.14%     | 0.8710 ± 0.0055    |
| 1.5         | 67.85% ± 3.19%     | 51.10% ± 6.31%     | 0.8786 ± 0.0061    |
| 2.0         | 69.22% ± 2.72%     | 53.65% ± 5.14%     | 0.8859 ± 0.0070    |
| **2.5**     | **69.64% ± 2.28%** | **54.62% ± 4.40%** | **0.8879 ± 0.0078**|
| 3.0         | 69.71% ± 2.09%     | 54.75% ± 4.09%     | 0.8877 ± 0.0084    |

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

### Temperature 调优

| Temperature | Subclass Overall   | Subclass Unseen    | Subclass AUROC     |
|-------------|--------------------|--------------------|--------------------|
| 0.0         | 38.68% ± 1.53%     | 48.09% ± 2.89%     | 0.8726 ± 0.0091    |
| **0.1**     | **65.94% ± 1.32%** | **47.96% ± 2.79%** | **0.8726 ± 0.0091**|
| 0.2         | 65.85% ± 1.30%     | 47.76% ± 2.77%     | 0.8725 ± 0.0091    |
| 0.4         | 65.61% ± 1.41%     | 47.22% ± 2.90%     | 0.8722 ± 0.0089    |
| 0.6         | 65.04% ± 1.77%     | 46.19% ± 3.52%     | 0.8714 ± 0.0088    |
| 0.8         | 64.84% ± 1.84%     | 45.95% ± 3.69%     | 0.8700 ± 0.0088    |
| 1.2         | 63.65% ± 2.26%     | 43.98% ± 4.50%     | 0.8647 ± 0.0091    |

观察：Energy-based 方法受益于较低温度，T=0.1 时性能最优。

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

# CAC 
## v1.0

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
class ExperimentConfig:
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
class ExperimentConfig:
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
class ExperimentConfig:
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

7 0 4
[Subclass] Overall            : 71.28% ± 0.96%
[Subclass] Seen               : 94.12% ± 0.59%
[Subclass] Unseen             : 51.81% ± 1.49%
AUROC                         : 0.8798 ± 0.0020
