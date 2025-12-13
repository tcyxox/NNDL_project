# Evaluations

## Baseline: 独立双头 + MSP (标准温度)

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
    enable_sigmoid_bce: bool = False  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 1  # OOD 温度缩放 (适用于 MSP 和 Energy)
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

## 独立双头 + MSP

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
    enable_sigmoid_bce: bool = False  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 3.5  # OOD 温度缩放 (适用于 MSP 和 Energy)
```

  [Superclass] Overall     : 95.00% ± 0.22%
  [Superclass] Seen        : 95.00% ± 0.22%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 64.46% ± 0.54%
  [Subclass] Seen          : 88.82% ± 0.37%
  [Subclass] Unseen        : 43.68% ± 0.88%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8719 ± 0.0008

观察：MSP 方法受益于较高温度，T=3.5 时性能最优。

## 独立双头 + MSP + Hierarchical Masking

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = False  # 训练时使用 SE Feature Gating
    enable_energy: bool = False  # 使用 Energy-based OOD 检测 替代 MSP
    enable_sigmoid_bce: bool = False  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 3.5  # OOD 温度缩放 (适用于 MSP 和 Energy)
```

  [Superclass] Overall     : 95.04% ± 0.17%       
  [Superclass] Seen        : 95.04% ± 0.17%       
  [Superclass] Unseen      : 0.00% ± 0.00%        
  [Subclass] Overall       : 65.23% ± 0.43%       
  [Subclass] Seen          : 89.33% ± 0.52%       
  [Subclass] Unseen        : 44.68% ± 1.13%       
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8760 ± 0.0010

结论：根据理论，Baseline + Hierarchical Masking >= Baseline 恒成立。

## Feature Gating 联合双头 + MSP + Hierarchical Masking

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = True  # 训练时使用 SE Feature Gating
    enable_energy: bool = False  # 使用 Energy-based OOD 检测 替代 MSP
    enable_sigmoid_bce: bool = False  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 3.5  # OOD 温度缩放 (适用于 MSP 和 Energy)
```

  [Superclass] Overall     : 95.07% ± 0.12%       
  [Superclass] Seen        : 95.07% ± 0.12%       
  [Superclass] Unseen      : 0.00% ± 0.00%        
  [Subclass] Overall       : 71.57% ± 1.76%       
  [Subclass] Seen          : 87.65% ± 0.97%       
  [Subclass] Unseen        : 57.86% ± 3.20%       
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8940 ± 0.0077

观察：开启 Feature Gating 后，需要使用更多 epochs 训练。

结论：Feature Gating 联合双头 能使未知subclass准确率显著提高。

## Feature Gating 联合双头 + Energy + Hierarchical Masking

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = True  # 训练时使用 SE Feature Gating
    enable_energy: bool = True  # 使用 Energy-based OOD 检测 替代 MSP
    enable_sigmoid_bce: bool = False  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 0.02  # OOD 温度缩放 (适用于 MSP 和 Energy)
```

观察：Energy-based 方法受益于较低温度，T=0.02 时性能最优。

  [Superclass] Overall     : 95.11% ± 0.11%       
  [Superclass] Seen        : 95.11% ± 0.11%       
  [Superclass] Unseen      : 0.00% ± 0.00%        
  [Subclass] Overall       : 66.03% ± 1.58%       
  [Subclass] Seen          : 87.02% ± 1.06%       
  [Subclass] Unseen        : 48.13% ± 3.26%       
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8780 ± 0.0083

?

  [Superclass] Overall     : 95.11% ± 0.11%
  [Superclass] Seen        : 95.11% ± 0.11%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 65.92% ± 1.57%
  [Subclass] Seen          : 87.02% ± 1.06%
  [Subclass] Unseen        : 47.93% ± 3.23%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8779 ± 0.0083

### 阶段性结论

1. Feature Gating 联合双头 + Hierarchical Masking 是一定要使用的。
2. MSP 受益于较高温度（T=3.5 时性能最优）：高温时，模型关注相对尖锐度。
3. Energy-based OOD 受益于较低温度（T=0.02 时性能最优）：低温时，模型关注绝对幅值。
4. 问题：Softmax 的强制归一化导致丢失了幅值信息；方案：使用基于 Logits 的 Energy-based OOD。
5. MSP 中使用基于 Softmax 的不保留幅值信息的阈值方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是统一的；而 Energy-based OOD 中使用基于 Logits 的保留幅值信息的阈值方法 + 基于 Softmax 的不保留幅值信息的 CE 损失函数，是不统一的。所以理论上应将 Softmax 替换为保留幅值信息的 Sigmoid。

# Feature Gating 联合双头 + Energy + Hierarchical Masking + Sigmoid & BCE

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_hierarchical_masking: bool = True  # 推理时使用 Hierarchical Masking
    enable_feature_gating: bool = True  # 训练时使用 SE Feature Gating
    enable_energy: bool = True  # 使用 Energy-based OOD 检测 替代 MSP
    enable_sigmoid_bce: bool = True  # 使用 Sigmoid + BCE 替代 Softmax + CE
    ood_temperature: float = 0.02  # OOD 温度缩放 (适用于 MSP 和 Energy)
```

  [Superclass] Overall     : 95.36% ± 0.23%
  [Superclass] Seen        : 95.36% ± 0.23%
  [Superclass] Unseen      : 0.00% ± 0.00%
  [Subclass] Overall       : 67.02% ± 0.34%
  [Subclass] Seen          : 88.27% ± 0.62%
  [Subclass] Unseen        : 48.90% ± 0.86%
  [Superclass] AUROC       : nan ± nan
  [Subclass] AUROC         : 0.8556 ± 0.0023

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

# OpenMax
## v1.0 LinearClassifier + OpenMax
```py
class ExperimentConfig:
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
class ExperimentConfig:
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
