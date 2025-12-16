import json
import os

import torch
import torch.nn.functional as F

from .models import LinearSingleHead, GatedDualHead
from .config import OODScoreMethod
from .scoring import compute_ood_score


def load_linear_single_head(prefix, model_dir, feature_dim, device):
    """
    Args:
        prefix: 'super' or 'sub'
        model_dir: 模型目录
        feature_dim: 特征维度
        device: 'cuda' or 'cpu'

    Returns:
        model: 加载好的模型
        local_to_global: 映射字典 (模型内部ID -> 原始ID)
    """
    # 1. 加载映射表 (Local ID -> Global ID)
    mapping_path = os.path.join(model_dir, f"{prefix}_local_to_global_map.json")
    with open(mapping_path, 'r') as f:
        local_to_global = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(local_to_global)
    print(f"[{prefix}] 加载映射表: 检测到 {num_classes} 个已知类")

    # 2. 初始化模型
    model = LinearSingleHead(feature_dim, num_classes)
    model_path = os.path.join(model_dir, f"{prefix}_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, local_to_global


def load_gated_dual_head(model_dir, feature_dim, num_super, num_sub, device):
    """
    Args:
        model_dir: 模型目录
        feature_dim: 特征维度
        num_super: 超类数量
        num_sub: 子类数量
        device: 'cuda' or 'cpu'

    Returns:
        model: 加载好的模型
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
    """

    # 加载映射表
    with open(os.path.join(model_dir, "super_local_to_global_map.json"), 'r') as f:
        super_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(model_dir, "sub_local_to_global_map.json"), 'r') as f:
        sub_map = {int(k): v for k, v in json.load(f).items()}

    print(f"[hierarchical] 加载映射表: {len(super_map)} 超类, {len(sub_map)} 子类")

    # 初始化并加载模型
    model = GatedDualHead(feature_dim, num_super, num_sub)
    model_path = os.path.join(model_dir, "hierarchical_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, super_map, sub_map


def _get_prediction_idx(logits):
    """
    获取预测类别索引（直接取 logits 的 argmax，等价于 argmax(softmax/sigmoid)）
    """
    return torch.argmax(logits, dim=1).item()


def _apply_hierarchical_masking(sub_logits, final_super, novel_super_idx, super_to_sub, 
                                 sub_global_to_local, num_sub_classes, device):
    """
    应用 Hierarchical Masking：mask 掉不属于预测超类的子类
    
    Returns:
        masked_sub_logits: 应用 mask 后的 sub logits
    """
    if final_super != novel_super_idx and final_super in super_to_sub:
        valid_subs = super_to_sub[final_super]
        mask = torch.full((1, num_sub_classes), float('-inf'), device=device)
        for sub_id in valid_subs:
            if sub_id in sub_global_to_local:
                local_id = sub_global_to_local[sub_id]
                mask[0, local_id] = 0
        return sub_logits + mask
    return sub_logits


def _predict_single_sample(
        super_logits, sub_logits, 
        super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx,
        super_to_sub, sub_global_to_local, num_sub_classes,
        temperature, score_method, device
):
    """
    预测单个样本的超类和子类
    
    Returns:
        final_super: 最终超类预测
        final_sub: 最终子类预测
        super_score: 超类 OOD 得分
        sub_score: 子类 OOD 得分
    """
    use_masking = (super_to_sub is not None)
    
    # === 超类预测 ===
    super_idx = _get_prediction_idx(super_logits)
    super_score = compute_ood_score(super_logits, temperature, score_method).item()
    is_novel_super = super_score < thresh_super
    final_super = novel_super_idx if is_novel_super else super_map[super_idx]
    
    # === 子类预测 ===
    # OOD 得分必须在 masking 之前计算
    sub_score = compute_ood_score(sub_logits, temperature, score_method).item()
    is_novel_sub = sub_score < thresh_sub
    
    # 应用 Hierarchical Masking
    if use_masking:
        sub_logits = _apply_hierarchical_masking(
            sub_logits, final_super, novel_super_idx, super_to_sub,
            sub_global_to_local, num_sub_classes, device
        )
    
    sub_idx = _get_prediction_idx(sub_logits)
    final_sub = novel_sub_idx if is_novel_sub else sub_map[sub_idx]
    
    # Hard Constraint: 超类 novel → 子类也 novel
    if final_super == novel_super_idx:
        final_sub = novel_sub_idx
    
    return final_super, final_sub, super_score, sub_score


def predict_with_linear_single_head(
        features, super_model, sub_model,
        super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx, device,
        super_to_sub, temperature, score_method: OODScoreMethod
):
    """
    使用两个独立的 LinearSingleHead 模型进行预测
    
    Returns:
        super_preds, sub_preds, super_scores, sub_scores
    """
    super_preds, sub_preds, super_scores, sub_scores = [], [], [], []
    
    use_masking = (super_to_sub is not None)
    num_sub_classes = sub_model.layer.out_features
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            super_logits = super_model(feature)
            sub_logits = sub_model(feature)
            
            final_super, final_sub, super_score, sub_score = _predict_single_sample(
                super_logits, sub_logits,
                super_map, sub_map,
                thresh_super, thresh_sub,
                novel_super_idx, novel_sub_idx,
                super_to_sub, sub_global_to_local, num_sub_classes,
                temperature, score_method, device
            )
            
            super_preds.append(final_super)
            sub_preds.append(final_sub)
            super_scores.append(super_score)
            sub_scores.append(sub_score)

    return super_preds, sub_preds, super_scores, sub_scores


def predict_with_gated_dual_head(
        features, model, super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx, device,
        super_to_sub, temperature, score_method: OODScoreMethod
):
    """
    使用 GatedDualHead 模型进行预测
    
    Returns:
        super_preds, sub_preds, super_scores, sub_scores
    """
    super_preds, sub_preds, super_scores, sub_scores = [], [], [], []
    
    use_masking = (super_to_sub is not None)
    num_sub_classes = len(sub_map)
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            super_logits, sub_logits = model(feature)
            
            final_super, final_sub, super_score, sub_score = _predict_single_sample(
                super_logits, sub_logits,
                super_map, sub_map,
                thresh_super, thresh_sub,
                novel_super_idx, novel_sub_idx,
                super_to_sub, sub_global_to_local, num_sub_classes,
                temperature, score_method, device
            )
            
            super_preds.append(final_super)
            sub_preds.append(final_sub)
            super_scores.append(super_score)
            sub_scores.append(sub_score)

    return super_preds, sub_preds, super_scores, sub_scores


def predict_with_openmax(
        features, model, openmax_super, openmax_sub,
        super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx, device,
        super_to_sub=None
):
    """
    使用 OpenMax + EER 阈值进行预测
    
    Args:
        features: 测试特征
        model: GatedDualHead 模型
        openmax_super: 超类 OpenMax 对象
        openmax_sub: 子类 OpenMax 对象
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
        thresh_super: 超类 unknown 概率阈值
        thresh_sub: 子类 unknown 概率阈值
        novel_super_idx: 超类 novel 标签
        novel_sub_idx: 子类 novel 标签
        device: 设备
        super_to_sub: 超类到子类的映射（用于 Hierarchical Masking）
        
    Returns:
        super_preds, sub_preds, super_unknown_probs, sub_unknown_probs
    """
    import numpy as np
    
    super_preds, sub_preds = [], []
    super_unknown_probs, sub_unknown_probs = [], []
    
    use_masking = (super_to_sub is not None)
    num_sub_classes = len(sub_map)
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    model.eval()
    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            super_logits, sub_logits = model(feature)
            
            # === 超类预测 ===
            super_probs = openmax_super.predict(super_logits)  # [1, num_super + 1]
            super_unknown_prob = super_probs[0, 0]
            super_unknown_probs.append(super_unknown_prob)
            
            if super_unknown_prob > thresh_super:
                final_super = novel_super_idx
            else:
                # 取 argmax (跳过第 0 列 unknown)
                super_known_probs = super_probs[0, 1:]
                super_idx = np.argmax(super_known_probs)
                final_super = super_map[super_idx]
            
            # === 子类预测 ===
            sub_probs = openmax_sub.predict(sub_logits)  # [1, num_sub + 1]
            sub_unknown_prob = sub_probs[0, 0]
            sub_unknown_probs.append(sub_unknown_prob)
            
            if sub_unknown_prob > thresh_sub:
                final_sub = novel_sub_idx
            else:
                # 取 argmax (跳过第 0 列 unknown)
                sub_known_probs = sub_probs[0, 1:]
                
                # 应用 Hierarchical Masking
                if use_masking and final_super != novel_super_idx and final_super in super_to_sub:
                    valid_subs = super_to_sub[final_super]
                    mask = np.full(num_sub_classes, float('-inf'))
                    for sub_id in valid_subs:
                        if sub_id in sub_global_to_local:
                            local_id = sub_global_to_local[sub_id]
                            mask[local_id] = 0
                    sub_known_probs = sub_known_probs + mask
                
                sub_idx = np.argmax(sub_known_probs)
                final_sub = sub_map[sub_idx]
            
            # Hard Constraint: 超类 novel → 子类也 novel
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx
            
            super_preds.append(final_super)
            sub_preds.append(final_sub)

    return super_preds, sub_preds, super_unknown_probs, sub_unknown_probs


