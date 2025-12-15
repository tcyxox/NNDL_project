"""
验证集相关函数 - 阈值计算

在验证集上计算 OSR 阈值，用于推理时判断未知类
"""
import torch

from .config import OODScoreMethod, ThresholdMethod
from .scoring import compute_ood_score, get_default_threshold



def calculate_threshold_linear_single_head(
        model, val_features, val_labels, label_map, device,
        use_full_val: bool,
        threshold_method: ThresholdMethod, target_recall, std_multiplier,
        temperature, score_method: OODScoreMethod
):
    """
    在验证集上为 LinearSingleHead 计算 OSR 阈值

    Args:
        model: LinearSingleHead 模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        device: 'cuda' or 'cpu'
        use_full_val: 是否使用完整验证集（已知+未知）计算阈值
        threshold_method: ThresholdMethod Enum - 阈值设定方法
        target_recall: 目标召回率 (如 0.95)，用于 Quantile 方法
        std_multiplier: 标准差乘数，用于 ZScore 方法
        temperature: OOD 温度缩放参数
        score_method: OODScoreMethod Enum - 得分计算方法

    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    if use_full_val:
        # 使用完整验证集（已知+未知）
        X_val = val_features.to(device)
    else:
        # 仅使用已知类样本
        known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
        X_val = val_features[known_mask].to(device)

    if len(X_val) == 0:
        print("警告: 验证集中没有样本，使用默认阈值")
        return get_default_threshold(score_method)

    with torch.no_grad():
        logits = model(X_val)
        scores = compute_ood_score(logits, temperature, score_method)
        
        if threshold_method == ThresholdMethod.ZScore:
            # 使用 mean - k*std 设定阈值
            threshold = (scores.mean() - std_multiplier * scores.std()).item()
        else:
            # 使用 target recall 设定阈值
            threshold = torch.quantile(scores, 1 - target_recall).item()

    return threshold


def calculate_threshold_gated_dual_head(
        model, val_features, val_super_labels, val_sub_labels,
        super_map_inv, sub_map_inv, device,
        use_full_val: bool,
        threshold_method: ThresholdMethod, target_recall, std_multiplier,
        temperature, score_method: OODScoreMethod
):
    """
    在验证集上为 GatedDualHead 计算 OSR 阈值

    Args:
        model: GatedDualHead 模型
        val_features: [N, D] - 验证集特征
        val_super_labels: [N] - 验证集超类标签
        val_sub_labels: [N] - 验证集子类标签
        super_map_inv: {local_id: global_id} - 超类映射
        sub_map_inv: {local_id: global_id} - 子类映射
        device: str - 'cuda' or 'cpu'
        use_full_val: bool - 是否使用完整验证集（已知+未知）计算阈值
        threshold_method: ThresholdMethod Enum - 阈值设定方法
        target_recall: float - 目标召回率 (如 0.95)，用于 Quantile 方法
        std_multiplier: float - 标准差乘数，用于 ZScore 方法
        temperature: float - OOD 温度缩放参数
        score_method: OODScoreMethod Enum - 得分计算方法

    Returns:
        thresh_super: float - 超类阈值
        thresh_sub: float - 子类阈值
    """

    def _calculate_single_threshold(logits, use_mask, threshold_method, target_recall, std_multiplier, temperature, score_method):
        if use_mask is None:
            # 使用完整验证集
            scores = compute_ood_score(logits, temperature, score_method)
        elif use_mask.sum() > 0:
            # 仅使用已知类样本
            scores = compute_ood_score(logits[use_mask], temperature, score_method)
        else:
            print(f"Warning: No known samples in validation set, using default threshold")
            return get_default_threshold(score_method)
        
        if threshold_method == ThresholdMethod.ZScore:
            # 使用 mean - k*std 设定阈值
            threshold = (scores.mean() - std_multiplier * scores.std()).item()
        else:
            # 使用 target recall 设定阈值
            threshold = torch.quantile(scores, 1 - target_recall).item()
        return threshold

    model.eval()

    with torch.no_grad():
        super_logits, sub_logits = model(val_features.to(device))

    if use_full_val:
        # 使用完整验证集
        thresh_super = _calculate_single_threshold(super_logits, None, threshold_method, target_recall, std_multiplier, temperature, score_method)
        thresh_sub = _calculate_single_threshold(sub_logits, None, threshold_method, target_recall, std_multiplier, temperature, score_method)
    else:
        # 仅使用已知类样本
        known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])
        thresh_super = _calculate_single_threshold(super_logits, known_super, threshold_method, target_recall, std_multiplier, temperature, score_method)

        known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])
        thresh_sub = _calculate_single_threshold(sub_logits, known_sub, threshold_method, target_recall, std_multiplier, temperature, score_method)

    return thresh_super, thresh_sub

