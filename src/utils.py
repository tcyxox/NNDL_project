import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from config import FEATURE_DIM


class LinearClassifier(nn.Module):
    """线性分类器模型"""
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


def load_mapping_and_model(prefix, model_dir, device):
    """
    加载 json 映射表和对应的模型
    
    Args:
        prefix: 'superclass' or 'subclass'
        model_dir: 模型目录
        device: 'cuda' or 'cpu'
    
    Returns:
        model: 加载好的模型
        local_to_global: 映射字典 (模型内部ID -> 原始ID)
    """
    # 1. 加载映射表 (Local ID -> Global ID)
    json_path = os.path.join(model_dir, f"{prefix}_mapping.json")
    with open(json_path, 'r') as f:
        local_to_global = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(local_to_global)
    print(f"[{prefix}] 加载映射表: 检测到 {num_classes} 个已知类")

    # 2. 初始化模型
    model = LinearClassifier(FEATURE_DIM, num_classes)
    model_path = os.path.join(model_dir, f"{prefix}_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, local_to_global


def calculate_threshold(model, val_features, val_labels, label_map, target_recall, device):
    """
    在验证集上计算 OSR 阈值
    
    Args:
        model: 分类模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        target_recall: 目标召回率 (如 0.95)
        device: 'cuda' or 'cpu'
    
    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    # 筛选出验证集里的"已知类"样本
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    if len(X_known) == 0:
        print("警告: 验证集中没有已知类样本，使用默认阈值 0.5")
        return 0.5

    with torch.no_grad():
        logits = model(X_known)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

    # 找到一个阈值 T，使得 target_recall% 的样本分数 > T
    threshold = torch.quantile(max_probs, 1 - target_recall).item()
    return threshold


def predict_with_osr(features, super_model, sub_model,
                     super_map, sub_map,
                     thresh_super, thresh_sub,
                     novel_super_idx, novel_sub_idx, device,
                     super_to_sub=None):
    """
    对特征进行 OSR 推理（支持 Hierarchical Masking）

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

            if max_super_prob.item() < thresh_super:
                final_super = novel_super_idx
            else:
                final_super = super_map[super_idx.item()]

            # === 子类预测（带 Hierarchical Masking）===
            sub_logits = sub_model(feature)
            
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
            max_sub_prob, sub_idx = torch.max(sub_probs, dim=1)

            if max_sub_prob.item() < thresh_sub:
                final_sub = novel_sub_idx
            else:
                final_sub = sub_map[sub_idx.item()]

            # === Hard Constraint: 超类 novel → 子类也 novel ===
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx

            super_preds.append(final_super)
            sub_preds.append(final_sub)

    return super_preds, sub_preds

