"""
验证集相关函数 - 阈值计算

在验证集上计算 OSR 阈值，用于推理时判断未知类
"""
import torch

from .config import OODScoreMethod
from .scoring import compute_ood_score, get_default_threshold



def calculate_threshold_linear_single_head(
        model, val_features, val_labels, label_map,
        target_recall, device, temperature, score_method: OODScoreMethod
):
    """
    在验证集上为 LinearSingleHead 计算 OSR 阈值

    Args:
        model: LinearSingleHead 模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        target_recall: 目标召回率 (如 0.95)
        device: 'cuda' or 'cpu'
        temperature: OOD 温度缩放参数
        score_method: OODScoreMethod ENUM - 得分计算方法

    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    if len(X_known) == 0:
        print("警告: 验证集中没有已知类样本，使用默认阈值")
        return get_default_threshold(score_method)

    with torch.no_grad():
        logits = model(X_known)
        scores = compute_ood_score(logits, temperature, score_method)
        threshold = torch.quantile(scores, 1 - target_recall).item()

    return threshold


def calculate_threshold_gated_dual_head(
        model, val_features, val_super_labels, val_sub_labels,
        super_map_inv, sub_map_inv, target_recall, device,
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
        target_recall: float - 目标召回率 (如 0.95)
        device: str - 'cuda' or 'cpu'
        temperature: float - OOD 温度缩放参数
        score_method: OODScoreMethod ENUM - 得分计算方法

    Returns:
        thresh_super: float - 超类阈值
        thresh_sub: float - 子类阈值
    """

    def _calculate_single_threshold(logits, known_mask, target_recall, temperature, score_method):
        if known_mask.sum() > 0:
            scores = compute_ood_score(logits[known_mask], temperature, score_method)
            threshold = torch.quantile(scores, 1 - target_recall).item()
        else:
            print(f"Warning: No known samples in validation set, using default threshold")
            threshold = get_default_threshold(score_method)
        return threshold

    model.eval()

    with torch.no_grad():
        super_logits, sub_logits = model(val_features.to(device))

    known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])
    thresh_super = _calculate_single_threshold(super_logits, known_super, target_recall, temperature, score_method)

    known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])
    thresh_sub = _calculate_single_threshold(sub_logits, known_sub, target_recall, temperature, score_method)

    return thresh_super, thresh_sub
