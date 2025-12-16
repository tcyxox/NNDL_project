import os
from dataclasses import dataclass, field
from enum import Enum

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)


# ================= Enum 定义 =================
class TrainingLoss(Enum):
    """训练损失函数类型"""
    CE = "ce"    # Cross Entropy (Softmax + CE)
    BCE = "bce"  # Binary Cross Entropy (Sigmoid + BCE)


class OODScoreMethod(Enum):
    """OOD 检测得分计算方法"""
    Energy = "energy"           # Energy-based score
    MSP = "msp"                 # Max Softmax Probability
    MaxSigmoid = "max_sigmoid"  # Max Sigmoid Probability


class KnownOnlyThreshold(Enum):
    """阈值设定方法 - 仅使用已知类样本"""
    Quantile = "quantile"  # 使用 target recall 设定阈值
    ZScore = "zscore"      # 使用 mean - k*std 设定阈值


class FullValThreshold(Enum):
    """阈值设定方法 - 需要已知+未知类样本"""
    EER = "eer"  # Equal Error Rate (已知/未知分布交叉点)


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
    # Outer Loop: Full -> Train (Pure Known) + Test (Mix)
    test_ratio: float = 0.2
    test_sub_novel_ratio: float = 0.1

    # Inner Loop: Train -> SubTrain (Pure Known) + Val (Mix)
    val_ratio: float = 0.2
    val_sub_novel_ratio: float = 0.1


@dataclass
class OSRConfig:
    novel_super_index: int = 3
    novel_sub_index: int = 87


@dataclass
class ModelConfig:
    # clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_model_id: str = "openai/clip-vit-large-patch14"  # 升级到 ViT-L/14
    # feature_dim: int = 512
    feature_dim: int = 768  # ViT-L/14 输出 768 维特征


@dataclass
class ExperimentConfig:
    seed: int = 42  # evaluate 时不使用，评估流程中只有 extract features 用到

    # 数据划分模式
    val_include_novel: bool = True  # True: val 不含未知类; False: val含未知类
    force_super_novel: bool = True   # 是否在 Val 中强制引入 Super Novel

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 75

    # 模型选择
    enable_hierarchical_masking: bool = True  # 推理时 Hierarchical Masking 开关
    enable_feature_gating: bool = True  # 训练时 SE Feature Gating 开关

    # 阈值设定（自动根据验证集是否有未知类选择方法）
    known_only_threshold: KnownOnlyThreshold = KnownOnlyThreshold.ZScore  # 无未知类时
    full_val_threshold: FullValThreshold = FullValThreshold.EER  # 有未知类时
    target_recall: float = 0.95  # Quantile 方法: target recall，95%
    std_multiplier: float = 1.645  # ZScore 方法: 标准差乘数，1.645

    # 方法选择
    training_loss: TrainingLoss = TrainingLoss.CE
    validation_score_method: OODScoreMethod = OODScoreMethod.MSP
    prediction_score_method: OODScoreMethod = OODScoreMethod.MSP
    validation_score_temperature: float = 1.5
    prediction_score_temperature: float = 1.5

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    osr: OSRConfig = field(default_factory=OSRConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


config = Config()

