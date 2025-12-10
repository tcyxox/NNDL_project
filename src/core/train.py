import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json

from .models import LinearClassifier, HierarchicalClassifier


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


def train_linear_model(features, labels, label_map, num_classes,
                       feature_dim, batch_size, learning_rate, epochs, device):
    """
    Args:
        features: 训练特征
        labels: 训练标签
        label_map: global_to_local 映射
        num_classes: 类别数量
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


def train_hierarchical_model(features, super_labels, sub_labels, super_map, sub_map, num_super, num_sub,
                             feature_dim, batch_size, learning_rate, epochs, device):
    """
    Args:
        features: 训练特征
        super_labels: 超类标签
        sub_labels: 子类标签
        num_super: 超类数量
        num_sub: 子类数量
        super_map: 超类 global_to_local 映射
        sub_map: 子类 global_to_local 映射
        feature_dim: 特征维度
        batch_size: 批大小
        learning_rate: 学习率
        epochs: 训练轮数
        device: 'cuda' or 'cpu'

    Returns:
        model: 训练好的模型
    """
    model = HierarchicalClassifier(feature_dim, num_super, num_sub)
    model.to(device)
    model.train()

    # 转换标签为 Local ID
    mapped_super = torch.tensor([super_map[l.item()] for l in super_labels], dtype=torch.long)
    mapped_sub = torch.tensor([sub_map[l.item()] for l in sub_labels], dtype=torch.long)

    # 创建数据集
    dataset = TensorDataset(features, mapped_super, mapped_sub)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, super_targets, sub_targets in loader:
            inputs = inputs.to(device)
            super_targets = super_targets.to(device)
            sub_targets = sub_targets.to(device)

            optimizer.zero_grad()
            super_logits, sub_logits = model(inputs)
            
            # 联合 Loss
            loss = criterion(super_logits, super_targets) + criterion(sub_logits, sub_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")

    return model


def run_training(feature_dir, output_dir, feature_dim, batch_size, learning_rate, 
                 epochs, enable_feature_gating, device):
    """
    训练模型的主函数
    
    Args:
        feature_dir: 特征目录
        output_dir: 输出目录
        feature_dim: 特征维度
        batch_size: 批大小
        learning_rate: 学习率
        epochs: 训练轮数
        enable_feature_gating: 是否启用 SE Feature Gating
        device: 'cuda' or 'cpu'
    """
    from .models import HierarchicalClassifier
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据
    print("正在加载训练数据...")
    train_features = torch.load(os.path.join(feature_dir, "train_features.pt"))
    train_super_labels = torch.load(os.path.join(feature_dir, "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))
    print(f"  > 训练样本数: {len(train_features)}")
    
    # 创建标签映射
    num_super, super_map = create_label_mapping(train_super_labels, "super", output_dir)
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", output_dir)
    
    # 训练模型
    if enable_feature_gating:
        print("\n=== 使用 Feature Gating (联合训练模式) ===")
        model = train_hierarchical_model(
            train_features, train_super_labels, train_sub_labels, super_map, sub_map, num_super, num_sub,
            feature_dim, batch_size, learning_rate, epochs, device
        )
        torch.save(model.state_dict(), os.path.join(output_dir, "hierarchical_model.pth"))
        print("层次模型已保存。")
    else:
        print("\n=== 独立训练模式 ===")
        print(f"开始训练 Superclass Model (Classes: {num_super})...")
        super_model = train_linear_model(
            train_features, train_super_labels, super_map, num_super,
            feature_dim, batch_size, learning_rate, epochs, device
        )
        torch.save(super_model.state_dict(), os.path.join(output_dir, "super_model.pth"))
        print("超类模型已保存。")
        
        print(f"开始训练 Subclass Model (Classes: {num_sub})...")
        sub_model = train_linear_model(
            train_features, train_sub_labels, sub_map, num_sub,
            feature_dim, batch_size, learning_rate, epochs, device
        )
        torch.save(sub_model.state_dict(), os.path.join(output_dir, "sub_model.pth"))
        print("子类模型已保存。")
    
    # 生成超类到子类的映射表
    create_super_to_sub_mapping(train_super_labels, train_sub_labels, output_dir)
    
    print("\n--- 所有模型训练完毕 ---")
    print(f"请检查 {output_dir} 目录下的模型文件 (.pth) 和 映射文件 (.json)。")
