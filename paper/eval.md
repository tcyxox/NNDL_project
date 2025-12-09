# Evaluations

## v1.0

### Config

```py
# ================= 模型配置 =================
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512

# ================= OSR 配置 =================
NOVEL_SUPER_INDEX = 3   # 未知超类的 ID
NOVEL_SUB_INDEX = 87    # 未知子类的 ID

# ================= 训练配置 =================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
TARGET_RECALL = 0.95    # 阈值计算时的目标召回率

# ================= 数据划分配置 =================
NOVEL_RATIO = 0.2       # 20% 子类作为未知类
TRAIN_RATIO = 0.8       # 已知类中 80% 用于训练
VAL_TEST_RATIO = 0.5    # Val 占 (Val+Test) 的比例
SEED = 42               # 随机种子
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

对任意super-sub pair使用hierarchical masking。其它同v1.0。

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
