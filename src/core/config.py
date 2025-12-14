import os
from dataclasses import dataclass, field
from enum import Enum

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)


# ================= ENUM 定义 =================
class TrainingLoss(Enum):
    """训练损失函数类型"""
    CE = "ce"    # Cross Entropy (Softmax + CE)
    BCE = "bce"  # Binary Cross Entropy (Sigmoid + BCE)


class OODScoreMethod(Enum):
    """OOD 检测得分计算方法"""
    Energy = "energy"           # Energy-based score
    MSP = "msp"                 # Max Softmax Probability
    MaxSigmoid = "max_sigmoid"  # Max Sigmoid Probability


# ================= 配置类 =================
@dataclass
class PathsConfig:
    data_raw: str = os.path.join(PROJECT_ROOT, "data/raw")
    data_processed: str = os.path.join(PROJECT_ROOT, "data/processed")
    features: str = os.path.join(PROJECT_ROOT, "data/processed/features")
    split_features: str = os.path.join(PROJECT_ROOT, "data/processed/split_features")
    split_images: str = os.path.join(PROJECT_ROOT, "data/processed/split_images")
    outputs: str = os.path.join(PROJECT_ROOT, "outputs")
    dev: str = os.path.join(PROJECT_ROOT, "outputs/dev")
    submit: str = os.path.join(PROJECT_ROOT, "outputs/submit")


@dataclass
class SplitConfig:
    novel_ratio: float = 0.2
    train_ratio: float = 0.8
    val_test_ratio: float = 0.5


@dataclass
class OSRConfig:
    novel_super_index: int = 3
    novel_sub_index: int = 87


@dataclass
class ModelConfig:
    clip_model_id: str = "openai/clip-vit-base-patch32"
    # clip_model_id: str = "openai/clip-vit-large-patch14"  # 升级到 ViT-L/14
    feature_dim: int = 512
    # feature_dim: int = 768  # ViT-L/14 输出 768 维特征


@dataclass
class ExperimentConfig:
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    target_recall: float = 0.95
    seed: int = 42

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    threshold_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_method: OODScoreMethod = OODScoreMethod.MSP

    # 温度参数
    threshold_temperature: float = 3.5
    prediction_temperature: float = 3.5


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    osr: OSRConfig = field(default_factory=OSRConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


config = Config()
