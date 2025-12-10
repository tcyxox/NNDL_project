import os

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

PATHS = {
    "data_raw": os.path.join(PROJECT_ROOT, "data/raw"),
    "data_processed": os.path.join(PROJECT_ROOT, "data/processed"),
    "features": os.path.join(PROJECT_ROOT, "data/processed/features"),
    "split": os.path.join(PROJECT_ROOT, "data/processed/split"),
    "outputs": os.path.join(PROJECT_ROOT, "outputs"),
    "dev": os.path.join(PROJECT_ROOT, "outputs/dev"),
    "submit": os.path.join(PROJECT_ROOT, "outputs/submit")
}

# ================= 模型配置 =================
MODEL = {
    "clip_model_id": "openai/clip-vit-base-patch32",
    "feature_dim": 512
}

# ================= OSR 配置 =================
OSR = {
    "novel_super_index": 3,
    "novel_sub_index": 87,
    "enable_hierarchical_masking": True
}

# ================= 训练配置 =================
TRAINING = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 50,
    "target_recall": 0.95,
    "seed": 42
}

# ================= 数据划分配置 =================
SPLIT = {
    "novel_ratio": 0.2,
    "train_ratio": 0.8,
    "val_test_ratio": 0.5
}
