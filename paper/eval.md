# Evaluations

## v1.0

### Config

```py
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42
    # 实验开关
    enable_soft_attention: bool = False
    enable_hierarchical_masking: bool = False
```

### Results

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
    enable_soft_attention: bool = False
    enable_hierarchical_masking: bool = True
```

  [Superclass] Overall     : 95.40% ± 0.16%
  [Superclass] Seen        : 95.40% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 61.37% ± 0.56%
  [Subclass] Seen          : 88.86% ± 0.15%
  [Subclass] Unseen        : 37.93% ± 1.02%

变化：未知subclass准确率降低，因为：
  如果超类被预测为已知类（如 superclass=1）
  → 子类 softmax 只在 ~24 个子类上分配概率
  → 概率更集中
  → 最大置信度变高
  → 更容易超过阈值，判为已知类
  为了从根本上解决这个问题，应该在logit层做阈值判断（不会被Masking影响到）

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
    enable_soft_attention: bool = True
    enable_hierarchical_masking: bool = True
```

  [Superclass] Overall     : 95.14% ± 0.16%
  [Superclass] Seen        : 95.14% ± 0.16%
  [Superclass] Unseen      :  0.00% ± 0.00%
  [Subclass] Overall       : 63.01% ± 1.17%
  [Subclass] Seen          : 88.16% ± 0.44%
  [Subclass] Unseen        : 41.57% ± 2.28%

变化：未知subclass准确率显著提高。
