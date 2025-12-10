# Evaluations

## v1.0

### Config

```py
@dataclass
class ModelConfig:
    clip_model_id: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 512


@dataclass
class OSRConfig:
    novel_super_index: int = 3
    novel_sub_index: int = 87
    enable_hierarchical_masking: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    target_recall: float = 0.95
    seed: int = 42


@dataclass
class SplitConfig:
    novel_ratio: float = 0.2
    train_ratio: float = 0.8
    val_test_ratio: float = 0.5
```

只当super是novel时，才强制sub也是novel。

### Results

[Superclass] 评估报告:
  > 总体准确率 (Overall): 95.31%
  > 已知类准确率 (Seen):    95.31%  (目标: 保持高)
  > 未知类准确率 (Unseen):  0.00%  (目标: >0, 越高越好)

[Subclass] 评估报告:
  > 总体准确率 (Overall): 62.55%
  > 已知类准确率 (Seen):    88.82%  (目标: 保持高)
  > 未知类准确率 (Unseen):  40.13%  (目标: >0, 越高越好)

## v1.1

```py
enable_hierarchical_masking: bool = True
```
其它同v1.0。

[Superclass] 评估报告:
  > 总体准确率 (Overall): 95.31%
  > 已知类准确率 (Seen):    95.31%  (目标: 保持高)
  > 未知类准确率 (Unseen):  0.00%  (目标: >0, 越高越好)

[Subclass] 评估报告:
  > 总体准确率 (Overall): 61.82%
  > 已知类准确率 (Seen):    88.82%  (目标: 保持高)
  > 未知类准确率 (Unseen):  38.80%  (目标: >0, 越高越好)

解读：
1. 已知subclass准确率提高或不变，符合预期。
2. 未知subclass准确率降低，因为：
  如果超类被预测为已知类（如 superclass=1）
  → 子类 softmax 只在 ~24 个子类上分配概率
  → 概率更集中
  → 最大置信度变高
  → 更容易超过阈值，判为已知类
  为了从根本上解决这个问题，应该在logit层做阈值判断（不会被Masking影响到）
