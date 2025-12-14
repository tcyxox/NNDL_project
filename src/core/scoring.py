"""
OOD 得分计算函数

用于计算样本的 OOD 得分，被 validation.py 和 prediction.py 共同使用
"""
import torch
import torch.nn.functional as F

from .config import OODScoreMethod


def compute_energy_score(logits, temperature):
    """
    计算 Energy Score: S(x; T) = T × log(Σ exp(logit_i / T))
    
    与原始 Energy 相比取负，使得：越高越可能是已知类（与 MSP/MaxSigmoid 方向一致）

    Args:
        logits: [N, C] - 模型输出的 logits
        temperature: float - 温度缩放参数

    Returns:
        scores: [N] - 能量得分，越高越可能是已知类
    """
    if temperature == 0:
        return torch.max(logits, dim=1)[0]
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_max_softmax_score(logits, temperature):
    """
    计算 Max Softmax Probability (MSP) Score
    
    Args:
        logits: [N, C] - 模型输出的 logits
        temperature: float - 温度缩放参数

    Returns:
        scores: [N] - MSP 得分，越高越可能是已知类
    """
    probs = F.softmax(logits / temperature, dim=1)
    return torch.max(probs, dim=1)[0]


def compute_max_sigmoid_score(logits, temperature):
    """
    计算 Max Sigmoid Score
    
    保留 logits 的幅值信息（不做归一化），适用于 BCE 训练的模型

    Args:
        logits: [N, C] - 模型输出的 logits
        temperature: float - 温度缩放参数

    Returns:
        scores: [N] - Max Sigmoid 得分，越高越可能是已知类
    """
    probs = torch.sigmoid(logits / temperature)
    return torch.max(probs, dim=1)[0]


def compute_ood_score(logits, temperature, score_method: OODScoreMethod):
    """
    统一的 OOD 得分计算接口
    
    根据 score_method 自动调用对应的计分函数，所有方法统一为：越高越可能是已知类

    Args:
        logits: [N, C] - 模型输出的 logits
        temperature: float - 温度缩放参数
        score_method: OODScoreMethod Enum - 得分计算方法

    Returns:
        scores: [N] - OOD 得分，越高越可能是已知类
    """
    if score_method == OODScoreMethod.Energy:
        return compute_energy_score(logits, temperature)
    elif score_method == OODScoreMethod.MaxSigmoid:
        return compute_max_sigmoid_score(logits, temperature)
    else:  # OODScoreMethod.MSP
        return compute_max_softmax_score(logits, temperature)


def get_default_threshold(score_method: OODScoreMethod):
    """
    获取默认阈值（当验证集中没有已知类样本时使用）
    """
    if score_method == OODScoreMethod.Energy:
        return 5
    else:
        return 0.5
