import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import random
import numpy as np

from .config import SEED
from .models import LinearClassifier


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


def create_label_mapping(labels, label_name, output_dir):
    """
    为标签创建连续ID映射，并保存映射文件

    Args:
        labels: 原始标签张量
        label_name: 标签名称 ('super' or 'sub')
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

    # 保存 local_to_global 映射关系，推理时必须用！
    mapping_path = os.path.join(output_dir, f"{label_name}_local_to_global_map.json")
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

    mapping_path = os.path.join(output_dir, "super_to_sub_map.json")
    with open(mapping_path, 'w') as f:
        json.dump(super_to_sub, f)
    print(f"  > 映射表已保存至: {mapping_path}")
    
    return super_to_sub
