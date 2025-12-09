import os

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 数据路径
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
FEATURES_DIR = os.path.join(DATA_PROCESSED_DIR, "features")
SPLIT_DIR = os.path.join(DATA_PROCESSED_DIR, "split")

# 输出路径
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")

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
