# Evaluations

## V1

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

### Results

```txt
[Superclass] 评估报告:
  > 总体准确率 (Overall): 94.94%
  > 已知类准确率 (Seen):    94.94%  (目标: 保持高)
  > 未知类准确率 (Unseen):  0.00%  (目标: >0, 越高越好)

[Subclass] 评估报告:
  > 总体准确率 (Overall): 66.64%
  > 已知类准确率 (Seen):    89.19%  (目标: 保持高)
  > 未知类准确率 (Unseen):  47.40%  (目标: >0, 越高越好)
```
