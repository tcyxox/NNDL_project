import os
from dataclasses import dataclass, field

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)


@dataclass
class PathsConfig:
    data_raw: str = os.path.join(PROJECT_ROOT, "data/raw")
    data_processed: str = os.path.join(PROJECT_ROOT, "data/processed")
    features: str = os.path.join(PROJECT_ROOT, "data/processed/features")
    split: str = os.path.join(PROJECT_ROOT, "data/processed/split")
    outputs: str = os.path.join(PROJECT_ROOT, "outputs")
    dev: str = os.path.join(PROJECT_ROOT, "outputs/dev")
    submit: str = os.path.join(PROJECT_ROOT, "outputs/submit")


@dataclass
class ModelConfig:
    clip_model_id: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 512


@dataclass
class OSRConfig:
    novel_super_index: int = 3
    novel_sub_index: int = 87
    enable_hierarchical_masking: bool = True


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


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    osr: OSRConfig = field(default_factory=OSRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


# 全局配置实例
config = Config()
