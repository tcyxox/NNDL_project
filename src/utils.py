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
