import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import random
import numpy as np

from config import FEATURE_DIM, SEED


def set_seed(seed=SEED):
    """设置所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


# =============================== 训练相关工具函数 ===============================

def create_label_mapping(labels, label_name, output_dir):
    """
    为标签创建连续ID映射，并保存映射文件

    Args:
        labels: 原始标签张量
        label_name: 标签名称 ('superclass' or 'subclass')
        output_dir: 输出目录

    Returns:
        num_classes: 类别数量
        global_to_local: 原始ID -> 模型内部ID 的映射
    """
    unique_classes = torch.unique(labels).sort()[0].tolist()
    num_classes = len(unique_classes)

    # 映射字典: 原始ID -> 模型内部ID (用于训练)
    global_to_local = {original: local for local, original in enumerate(unique_classes)}
    # 反向字典: 模型内部ID -> 原始ID (用于推理恢复)
    local_to_global = {local: original for local, original in enumerate(unique_classes)}

    print(f"[{label_name}] 检测到 {num_classes} 个已知类别。")
    print(f"  > 原始标签示例: {unique_classes[:5]}...")

    # 保存映射关系，推理时必须用！
    mapping_path = os.path.join(output_dir, f"{label_name}_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(local_to_global, f)
    print(f"  > 映射表已保存至: {mapping_path}")

    return num_classes, global_to_local


def train_classifier(features, labels, label_map, num_classes, model_name, 
                     feature_dim, batch_size, learning_rate, epochs, device):
    """
    训练分类器模型

    Args:
        features: 训练特征
        labels: 训练标签
        label_map: global_to_local 映射
        num_classes: 类别数量
        model_name: 模型名称（用于日志）
        feature_dim: 特征维度
        batch_size: 批大小
        learning_rate: 学习率
        epochs: 训练轮数
        device: 'cuda' or 'cpu'

    Returns:
        model: 训练好的模型
    """
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    # 初始化模型
    model = LinearClassifier(feature_dim, num_classes)
    model.to(device)
    model.train()

    # 将所有标签转换为 Local ID
    mapped_labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)

    # 创建数据集
    dataset = TensorDataset(features, mapped_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n开始训练 {model_name} (Classes: {num_classes})...")

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")

    return model


def create_super_to_sub_mapping(super_labels, sub_labels, output_dir):
    """
    生成并保存超类到子类的映射表
    
    Args:
        super_labels: 超类标签
        sub_labels: 子类标签
        output_dir: 输出目录
    
    Returns:
        super_to_sub: 映射字典 {super_id: [sub_ids]}
    """
    print("\n生成超类到子类映射表...")
    super_to_sub = {}
    unique_super = torch.unique(super_labels).tolist()
    for super_idx in unique_super:
        mask = (super_labels == super_idx)
        sub_indices = torch.unique(sub_labels[mask]).tolist()
        super_to_sub[super_idx] = sub_indices
        print(f"  > Superclass {super_idx}: {len(sub_indices)} subclasses")

    mapping_path = os.path.join(output_dir, "super_to_sub_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(super_to_sub, f)
    print(f"  > 映射表已保存至: {mapping_path}")
    
    return super_to_sub


