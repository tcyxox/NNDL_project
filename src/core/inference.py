import json
import os

import torch
import torch.nn.functional as F

from .models import LinearClassifier, HierarchicalClassifier


def compute_energy(logits):
    """
    计算 Energy Score: E(x) = -log Σ exp(logit_i)
    能量越低越可能是已知类，能量越高越可能是未知类
    """
    return -torch.logsumexp(logits, dim=1)


def calculate_threshold_linear(model, val_features, val_labels, label_map, target_recall, device, use_energy):
    """
    在验证集上为 LinearClassifier 计算 OSR 阈值

    Args:
        model: LinearClassifier 模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        target_recall: 目标召回率 (如 0.95)
        device: 'cuda' or 'cpu'
        use_energy: 是否使用 energy score（否则用 MSP）

    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    # 筛选出验证集里的"已知类"样本
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    if len(X_known) == 0:
        print("警告: 验证集中没有已知类样本，使用默认阈值")
        return -5 if use_energy else 0.5

    with torch.no_grad():
        logits = model(X_known)
        if use_energy:
            # Energy: 越低越可能是已知类，阈值是能量上限
            scores = compute_energy(logits)
            threshold = torch.quantile(scores, target_recall).item()
        else:
            # MSP: 越高越可能是已知类，阈值是概率下限
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            threshold = torch.quantile(max_probs, 1 - target_recall).item()

    return threshold


def calculate_threshold_hierarchical(model, val_features, val_super_labels, val_sub_labels, 
                                      super_map_inv, sub_map_inv, target_recall, device, use_energy):
    """
    在验证集上为 HierarchicalClassifier 计算 OSR 阈值

    Args:
        model: HierarchicalClassifier 模型
        val_features: 验证集特征
        val_super_labels: 验证集超类标签
        val_sub_labels: 验证集子类标签
        super_map_inv: 超类 local_to_global 映射
        sub_map_inv: 子类 local_to_global 映射
        target_recall: 目标召回率 (如 0.95)
        device: 'cuda' or 'cpu'
        use_energy: 是否使用 energy score（否则用 MSP）

    Returns:
        thresh_super: 超类阈值
        thresh_sub: 子类阈值
    """
    model.eval()
    
    with torch.no_grad():
        super_logits, sub_logits = model(val_features.to(device))
    
    # 超类阈值
    known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])
    if known_super.sum() > 0:
        if use_energy:
            scores = compute_energy(super_logits[known_super])
            thresh_super = torch.quantile(scores, target_recall).item()
        else:
            super_probs = F.softmax(super_logits[known_super], dim=1)
            thresh_super = torch.quantile(super_probs.max(dim=1)[0], 1 - target_recall).item()
    else:
        print("警告: 验证集中没有已知超类样本，使用默认阈值")
        thresh_super = -5 if use_energy else 0.5
    
    # 子类阈值
    known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])
    if known_sub.sum() > 0:
        if use_energy:
            scores = compute_energy(sub_logits[known_sub])
            thresh_sub = torch.quantile(scores, target_recall).item()
        else:
            sub_probs = F.softmax(sub_logits[known_sub], dim=1)
            thresh_sub = torch.quantile(sub_probs.max(dim=1)[0], 1 - target_recall).item()
    else:
        print("警告: 验证集中没有已知子类样本，使用默认阈值")
        thresh_sub = -5 if use_energy else 0.5
    
    return thresh_super, thresh_sub


def load_linear_model(prefix, model_dir, feature_dim, device):
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
    model = LinearClassifier(feature_dim, num_classes)
    model_path = os.path.join(model_dir, f"{prefix}_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, local_to_global


def predict_with_linear_model(features, super_model, sub_model,
                              super_map, sub_map,
                              thresh_super, thresh_sub,
                              novel_super_idx, novel_sub_idx, device,
                              use_energy, super_to_sub):
    """
    Args:
        features: 输入特征 [N, 512]
        super_model: 超类分类模型
        sub_model: 子类分类模型
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
        thresh_super: 超类阈值
        thresh_sub: 子类阈值
        novel_super_idx: 未知超类的 ID (3)
        novel_sub_idx: 未知子类的 ID (87)
        device: 'cuda' or 'cpu'
        super_to_sub: 超类到子类的映射 {super_id: [sub_ids]}（可选，用于 masking）
        use_energy: 是否使用 energy score（否则用 MSP）

    Returns:
        super_preds: 超类预测列表
        sub_preds: 子类预测列表
    """
    super_preds = []
    sub_preds = []
    
    # 是否启用 hierarchical masking
    use_masking = (super_to_sub is not None)
    num_sub_classes = sub_model.layer.out_features
    
    # 如果启用 masking，从 sub_map 反转计算 global_to_local
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)

            # === 超类预测 ===
            super_logits = super_model(feature)
            super_probs = F.softmax(super_logits, dim=1)
            max_super_prob, super_idx = torch.max(super_probs, dim=1)
            
            # 判断是否为 novel
            if use_energy:
                energy = compute_energy(super_logits).item()
                is_novel_super = energy > thresh_super
            else:
                is_novel_super = max_super_prob.item() < thresh_super

            if is_novel_super:
                final_super = novel_super_idx
            else:
                final_super = super_map[super_idx.item()]

            # === 子类预测（带 Hierarchical Masking）===
            sub_logits = sub_model(feature)
            
            # OOD 分数需要在 masking 之前计算（与阈值计算保持一致）
            if use_energy:
                sub_score = compute_energy(sub_logits).item()
            else:
                sub_probs_unmasked = F.softmax(sub_logits, dim=1)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()
            
            # 如果超类不是 novel 且启用了 masking，则 mask 掉不属于该超类的子类
            if use_masking and final_super != novel_super_idx and final_super in super_to_sub:
                valid_subs = super_to_sub[final_super]
                mask = torch.full((1, num_sub_classes), float('-inf'), device=device)
                for sub_id in valid_subs:
                    if sub_id in sub_global_to_local:
                        local_id = sub_global_to_local[sub_id]
                        mask[0, local_id] = 0
                sub_logits = sub_logits + mask
            
            sub_probs = F.softmax(sub_logits, dim=1)
            _, sub_idx = torch.max(sub_probs, dim=1)
            
            # 判断是否为 novel（使用 masking 前计算的分数）
            if use_energy:
                is_novel_sub = sub_score > thresh_sub
            else:
                is_novel_sub = sub_score < thresh_sub

            if is_novel_sub:
                final_sub = novel_sub_idx
            else:
                final_sub = sub_map[sub_idx.item()]

            # === Hard Constraint: 超类 novel → 子类也 novel ===
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx

            super_preds.append(final_super)
            sub_preds.append(final_sub)

    return super_preds, sub_preds


def load_hierarchical_model(model_dir, feature_dim, num_super, num_sub, device):
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
    model = HierarchicalClassifier(feature_dim, num_super, num_sub)
    model_path = os.path.join(model_dir, "hierarchical_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    return model, super_map, sub_map


def predict_with_hierarchical_model(features, model, super_map, sub_map,
                                    thresh_super, thresh_sub,
                                    novel_super_idx, novel_sub_idx, device,
                                    use_energy, super_to_sub):
    """
    Args:
        features: 输入特征 [N, 512]
        model: HierarchicalClassifier 模型
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
        thresh_super: 超类阈值
        thresh_sub: 子类阈值
        novel_super_idx: 未知超类的 ID (3)
        novel_sub_idx: 未知子类的 ID (87)
        device: 'cuda' or 'cpu'
        super_to_sub: 超类到子类的映射（用于 hard masking，可选）
        use_energy: 是否使用 energy score（否则用 MSP）
    
    Returns:
        super_preds: 超类预测列表
        sub_preds: 子类预测列表
    """
    super_preds = []
    sub_preds = []
    
    use_masking = (super_to_sub is not None)
    num_sub_classes = len(sub_map)
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None
    
    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            
            # 模型同时输出 super 和 sub logits
            super_logits, sub_logits = model(feature)
            
            # === 超类预测 ===
            super_probs = F.softmax(super_logits, dim=1)
            max_super_prob, super_idx = torch.max(super_probs, dim=1)
            
            # 判断是否为 novel
            if use_energy:
                energy = compute_energy(super_logits).item()
                is_novel_super = energy > thresh_super
            else:
                is_novel_super = max_super_prob.item() < thresh_super
            
            if is_novel_super:
                final_super = novel_super_idx
            else:
                final_super = super_map[super_idx.item()]
            
            # === 子类预测（可选 Hard Masking）===
            # OOD 分数需要在 masking 之前计算（与阈值计算保持一致）
            if use_energy:
                sub_score = compute_energy(sub_logits).item()
            else:
                sub_probs_unmasked = F.softmax(sub_logits, dim=1)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()
            
            if use_masking and final_super != novel_super_idx and final_super in super_to_sub:
                valid_subs = super_to_sub[final_super]
                mask = torch.full((1, num_sub_classes), float('-inf'), device=device)
                for sub_id in valid_subs:
                    if sub_id in sub_global_to_local:
                        local_id = sub_global_to_local[sub_id]
                        mask[0, local_id] = 0
                sub_logits = sub_logits + mask
            
            sub_probs = F.softmax(sub_logits, dim=1)
            _, sub_idx = torch.max(sub_probs, dim=1)
            
            # 判断是否为 novel（使用 masking 前计算的分数）
            if use_energy:
                is_novel_sub = sub_score > thresh_sub
            else:
                is_novel_sub = sub_score < thresh_sub
            
            if is_novel_sub:
                final_sub = novel_sub_idx
            else:
                final_sub = sub_map[sub_idx.item()]
            
            # === Hard Constraint: 超类 novel → 子类也 novel ===
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx
            
            super_preds.append(final_super)
            sub_preds.append(final_sub)
    
    return super_preds, sub_preds
