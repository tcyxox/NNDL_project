# Evaluations

## v1.0

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

  [Superclass] Overall     : 95.40% ± 0.16%
  [Superclass] Seen        : 95.40% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 62.17% ± 0.63%
  [Subclass] Seen          : 88.82% ± 0.21%
  [Subclass] Unseen        : 39.43% ± 1.16%

## v1.1

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

  [Superclass] Overall     : 95.40% ± 0.16%
  [Superclass] Seen        : 95.40% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 62.17% ± 0.63%
  [Subclass] Seen          : 88.82% ± 0.21%
  [Subclass] Unseen        : 39.43% ± 1.16%

## v1.2

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

  [Superclass] Overall     : 95.34% ± 0.17%
  [Superclass] Seen        : 95.34% ± 0.17%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 65.90% ± 2.39%
  [Subclass] Seen          : 87.73% ± 0.92%
  [Subclass] Unseen        : 47.29% ± 4.50%

变化：未知subclass准确率显著提高。

## v1.3

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

  [Superclass] Overall     : 95.22% ± 0.20%
  [Superclass] Seen        : 95.22% ± 0.20%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 64.31% ± 2.08%
  [Subclass] Seen          : 86.71% ± 0.67%
  [Subclass] Unseen        : 45.22% ± 4.21%