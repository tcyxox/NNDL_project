"""
验证集相关函数 - 阈值计算

在验证集上计算 OSR 阈值，用于推理时判断未知类
自动根据验证集是否包含未知类样本选择阈值方法
"""
import torch

from .config import OODScoreMethod, KnownOnlyThreshold, FullValThreshold
from .scoring import compute_ood_score, get_default_threshold


def calculate_threshold_linear_single_head(
        model, val_features, val_labels, label_map, device,
        test_only_unknown: bool,
        known_only_method: KnownOnlyThreshold,
        full_val_method: FullValThreshold,
        target_recall, std_multiplier,
        temperature, score_method: OODScoreMethod
):
    """
    在验证集上为 LinearSingleHead 计算 OSR 阈值
    根据 test_only_unknown 选择阈值方法

    Args:
        model: LinearSingleHead 模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        device: 'cuda' or 'cpu'
        test_only_unknown: 是否仅 test 含未知类（决定使用哪种阈值方法）
        known_only_method: KnownOnlyThreshold Enum - 无未知类时使用的方法
        full_val_method: FullValThreshold Enum - 有未知类时使用的方法
        target_recall: 目标召回率 (如 0.95)，用于 Quantile 方法
        std_multiplier: 标准差乘数，用于 ZScore 方法
        temperature: OOD 温度缩放参数
        score_method: OODScoreMethod Enum - 得分计算方法

    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    # 计算 known/unknown mask
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    unknown_mask = ~known_mask

    with torch.no_grad():
        all_logits = model(val_features.to(device))
        all_scores = compute_ood_score(all_logits, temperature, score_method)

    known_scores = all_scores[known_mask]
    unknown_scores = all_scores[unknown_mask]
    has_unknown = unknown_mask.sum() > 0

    # 根据 test_only_unknown 决定使用哪种方法，并添加 fallback 逻辑
    if test_only_unknown:
        # Val 不含未知类，使用 KnownOnly 方法
        threshold = _apply_known_only_threshold(
            known_only_method, known_scores, target_recall, std_multiplier, score_method
        )
    else:
        # Val 应含未知类
        if has_unknown:
            threshold = _apply_full_val_threshold(
                full_val_method, known_scores, unknown_scores, score_method
            )
        else:
            # Fallback: 配置期望有未知类，但实际数据没有，回退到 KnownOnly
            print("警告: test_only_unknown=False 但 Val 中无未知样本，回退到 KnownOnly 方法")
            threshold = _apply_known_only_threshold(
                known_only_method, known_scores, target_recall, std_multiplier, score_method
            )

    return threshold


def calculate_threshold_gated_dual_head(
        model, val_features, val_super_labels, val_sub_labels,
        super_map_inv, sub_map_inv, device,
        test_only_unknown: bool,
        known_only_method: KnownOnlyThreshold,
        full_val_method: FullValThreshold,
        target_recall, std_multiplier,
        temperature, score_method: OODScoreMethod
):
    """
    在验证集上为 GatedDualHead 计算 OSR 阈值
    根据 test_only_unknown 选择阈值方法

    Args:
        model: GatedDualHead 模型
        val_features: [N, D] - 验证集特征
        val_super_labels: [N] - 验证集超类标签
        val_sub_labels: [N] - 验证集子类标签
        super_map_inv: {local_id: global_id} - 超类映射
        sub_map_inv: {local_id: global_id} - 子类映射
        device: str - 'cuda' or 'cpu'
        test_only_unknown: 是否仅 test 含未知类（决定使用哪种阈值方法）
        known_only_method: KnownOnlyThreshold Enum - 无未知类时使用的方法
        full_val_method: FullValThreshold Enum - 有未知类时使用的方法
        target_recall: float - 目标召回率 (如 0.95)，用于 Quantile 方法
        std_multiplier: float - 标准差乘数，用于 ZScore 方法
        temperature: float - OOD 温度缩放参数
        score_method: OODScoreMethod Enum - 得分计算方法

    Returns:
        thresh_super: float - 超类阈值
        thresh_sub: float - 子类阈值
    """
    model.eval()

    # 计算 known/unknown masks
    known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])
    known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])
    unknown_super = ~known_super
    unknown_sub = ~known_sub

    with torch.no_grad():
        super_logits, sub_logits = model(val_features.to(device))
        super_scores = compute_ood_score(super_logits, temperature, score_method)
        sub_scores = compute_ood_score(sub_logits, temperature, score_method)

    known_super_scores = super_scores[known_super]
    known_sub_scores = sub_scores[known_sub]
    unknown_super_scores = super_scores[unknown_super]
    unknown_sub_scores = sub_scores[unknown_sub]
    has_unknown_super = unknown_super.sum() > 0
    has_unknown_sub = unknown_sub.sum() > 0

    # 根据 test_only_unknown 决定使用哪种方法，并添加 fallback 逻辑
    if test_only_unknown:
        # Val 不含未知类，使用 KnownOnly 方法
        thresh_super = _apply_known_only_threshold(
            known_only_method, known_super_scores, target_recall, std_multiplier, score_method
        )
        thresh_sub = _apply_known_only_threshold(
            known_only_method, known_sub_scores, target_recall, std_multiplier, score_method
        )
    else:
        # Val 应含未知类
        # Superclass threshold (with fallback)
        if has_unknown_super:
            thresh_super = _apply_full_val_threshold(
                full_val_method, known_super_scores, unknown_super_scores, score_method
            )
        else:
            print("警告: test_only_unknown=False 但 Val 中无未知超类，回退到 KnownOnly 方法")
            thresh_super = _apply_known_only_threshold(
                known_only_method, known_super_scores, target_recall, std_multiplier, score_method
            )
        
        # Subclass threshold (with fallback)
        if has_unknown_sub:
            thresh_sub = _apply_full_val_threshold(
                full_val_method, known_sub_scores, unknown_sub_scores, score_method
            )
        else:
            print("警告: test_only_unknown=False 但 Val 中无未知子类，回退到 KnownOnly 方法")
            thresh_sub = _apply_known_only_threshold(
                known_only_method, known_sub_scores, target_recall, std_multiplier, score_method
            )

    return thresh_super, thresh_sub


def _apply_known_only_threshold(
        method: KnownOnlyThreshold,
        known_scores: torch.Tensor,
        target_recall: float,
        std_multiplier: float,
        score_method: OODScoreMethod
) -> float:
    """
    应用仅使用已知类样本的阈值方法
    """
    if len(known_scores) == 0:
        print("警告: 已知类样本为空，使用默认阈值")
        return get_default_threshold(score_method)

    if method == KnownOnlyThreshold.ZScore:
        return (known_scores.mean() - std_multiplier * known_scores.std()).item()
    else:  # Quantile
        return torch.quantile(known_scores, 1 - target_recall).item()


def _apply_full_val_threshold(
        method: FullValThreshold,
        known_scores: torch.Tensor,
        unknown_scores: torch.Tensor,
        score_method: OODScoreMethod
) -> float:
    """
    应用需要未知类样本的阈值方法
    """
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        print("警告: 已知或未知样本为空，使用默认阈值")
        return get_default_threshold(score_method)

    if method == FullValThreshold.Intersection:
        return find_distribution_intersection(known_scores, unknown_scores)
    
    # 未来可以添加更多方法
    return get_default_threshold(score_method)


def find_distribution_intersection(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    """
    找到已知类和未知类分数分布的交叉点
    
    Args:
        known_scores: 已知类样本的 OOD 分数（通常较高）
        unknown_scores: 未知类样本的 OOD 分数（通常较低）
    
    Returns:
        threshold: 两个分布的交叉点阈值
    """
    # 合并所有分数作为候选阈值
    all_scores = torch.cat([known_scores, unknown_scores])
    candidates = torch.sort(all_scores)[0]
    
    # 对每个候选阈值，计算两个分布在该点的"密度差"
    # 使用经验 CDF 的差异来近似
    best_threshold = candidates[len(candidates) // 2].item()  # 默认中间值
    min_diff = float('inf')
    
    for t in candidates:
        t_val = t.item()
        # 已知类中低于阈值的比例（应该尽量小）
        known_below = (known_scores < t_val).float().mean().item()
        # 未知类中低于阈值的比例（应该尽量大）
        unknown_below = (unknown_scores < t_val).float().mean().item()
        
        # 交叉点：两个比例相等的地方
        # 即 P(known < t) = P(unknown >= t) = 1 - P(unknown < t)
        # 也就是 known_below = 1 - unknown_below
        diff = abs(known_below - (1 - unknown_below))
        
        if diff < min_diff:
            min_diff = diff
            best_threshold = t_val
    
    return best_threshold
