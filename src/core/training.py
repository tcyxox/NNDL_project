import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .models import LinearSingleHead, GatedDualHead
from .config import TrainingLoss


def create_label_mapping(labels, label_name, output_dir=None, verbose=True):
    """
    为标签创建连续ID映射

    Args:
        labels: 原始标签张量
        label_name: 标签名称 ('super' or 'sub')
        output_dir: 输出目录（可选，如果提供则保存映射文件）

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

    if verbose:
        print(f"[{label_name}] 检测到 {num_classes} 个已知类别。")
        print(f"  > 原始标签示例: {unique_classes[:5]}...")

    # 保存 local_to_global 映射关系（可选）
    if output_dir:
        mapping_path = os.path.join(output_dir, f"{label_name}_local_to_global_map.json")
        with open(mapping_path, 'w') as f:
            json.dump(local_to_global, f)
        if verbose:
            print(f"  > 映射表已保存至: {mapping_path}")

    return num_classes, global_to_local


def create_super_to_sub_mapping(super_labels, sub_labels, output_dir=None, verbose=True):
    """
    生成超类到子类的映射表

    Args:
        super_labels: 超类标签
        sub_labels: 子类标签
        output_dir: 输出目录（可选，如果提供则保存映射文件）

    Returns:
        super_to_sub: 映射字典 {super_id: [sub_ids]}
    """
    if verbose:
        print("\n生成超类到子类映射表...")
    super_to_sub = {}
    unique_super = torch.unique(super_labels).tolist()
    for super_idx in unique_super:
        mask = (super_labels == super_idx)
        sub_indices = torch.unique(sub_labels[mask]).tolist()
        super_to_sub[super_idx] = sub_indices
        if verbose:
            print(f"  > Superclass {super_idx}: {len(sub_indices)} subclasses")

    if output_dir:
        mapping_path = os.path.join(output_dir, "super_to_sub_map.json")
        with open(mapping_path, 'w') as f:
            json.dump(super_to_sub, f)
        if verbose:
            print(f"  > 映射表已保存至: {mapping_path}")

    return super_to_sub


def train_linear_single_head(features, labels, label_map, num_classes,
                             feature_dim, batch_size, learning_rate, epochs, device, training_loss, verbose=True):
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
        training_loss: TrainingLoss Enum - 训练损失函数类型

    Returns:
        model: 训练好的模型
    """
    # 初始化模型
    model = LinearSingleHead(feature_dim, num_classes)
    model.to(device)
    model.train()

    # 将所有标签转换为 Local ID
    mapped_labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)

    # 创建数据集
    dataset = TensorDataset(features, mapped_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 选择损失函数
    if training_loss == TrainingLoss.BCE:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if training_loss == TrainingLoss.BCE:
                # 将标签转换为 one-hot 格式
                targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_onehot)
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")

    return model


def train_gated_dual_head(features, super_labels, sub_labels, super_map, sub_map, num_super, num_sub,
                          feature_dim, batch_size, learning_rate, epochs, device, training_loss, verbose=True):
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
        training_loss: TrainingLoss Enum - 训练损失函数类型

    Returns:
        model: 训练好的模型
    """
    model = GatedDualHead(feature_dim, num_super, num_sub)
    model.to(device)
    model.train()

    # 转换标签为 Local ID
    mapped_super = torch.tensor([super_map[l.item()] for l in super_labels], dtype=torch.long)
    mapped_sub = torch.tensor([sub_map[l.item()] for l in sub_labels], dtype=torch.long)

    # 创建数据集
    dataset = TensorDataset(features, mapped_super, mapped_sub)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 选择损失函数
    if training_loss == TrainingLoss.BCE:
        criterion = nn.BCEWithLogitsLoss()
    else:
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
            if training_loss == TrainingLoss.BCE:
                # 将标签转换为 one-hot 格式
                super_onehot = torch.nn.functional.one_hot(super_targets, num_classes=num_super).float()
                sub_onehot = torch.nn.functional.one_hot(sub_targets, num_classes=num_sub).float()
                loss = criterion(super_logits, super_onehot) + criterion(sub_logits, sub_onehot)
            else:
                loss = criterion(super_logits, super_targets) + criterion(sub_logits, sub_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")

    return model


def run_training(feature_dim, batch_size, learning_rate, epochs, device,
                 enable_feature_gating, training_loss,
                 train_features=None, train_super_labels=None, train_sub_labels=None,
                 feature_dir=None, output_dir=None, verbose=True):
    """
    训练模型的主函数

    Args:
        feature_dim: 特征维度
        batch_size: 批大小
        learning_rate: 学习率
        epochs: 训练轮数
        device: 'cuda' or 'cpu'
        enable_feature_gating: 是否启用 SE Feature Gating
        training_loss: TrainingLoss Enum - 训练损失函数类型
        train_features: 训练特征（可选，若提供则直接使用）
        train_super_labels: 训练超类标签（可选）
        train_sub_labels: 训练子类标签（可选）
        feature_dir: 特征目录（可选，若未提供 train_features 则从此加载）
        output_dir: 输出目录（可选，如果提供则保存模型和映射文件）

    Returns:
        如果 enable_feature_gating=True: (model, super_map, sub_map, super_to_sub)
        如果 enable_feature_gating=False: (super_model, sub_model, super_map, sub_map, super_to_sub)
    """

    # 加载或使用传入的训练数据
    if train_features is None:
        if feature_dir is None:
            raise ValueError("必须提供 train_features 或 feature_dir")
        if verbose:
            print("正在加载训练数据...")
        train_features = torch.load(os.path.join(feature_dir, "train_features.pt"))
        train_super_labels = torch.load(os.path.join(feature_dir, "train_super_labels.pt"))
        train_sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))
    
    if verbose:
        print(f"  > 训练样本数: {len(train_features)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 创建标签映射
    num_super, super_map = create_label_mapping(train_super_labels, "super", output_dir, verbose)
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", output_dir, verbose)

    # 训练模型
    if enable_feature_gating:
        if verbose:
            print(f"\n=== 使用 Feature Gating (联合训练模式) | Loss: {training_loss.value.upper()} ===")
        model = train_gated_dual_head(
            train_features, train_super_labels, train_sub_labels, super_map, sub_map, num_super, num_sub,
            feature_dim, batch_size, learning_rate, epochs, device, training_loss, verbose
        )
        if output_dir:
            torch.save(model.state_dict(), os.path.join(output_dir, "hierarchical_model.pth"))
            if verbose:
                print("层次模型已保存。")
    else:
        if verbose:
            print("\n=== 独立训练模式 ===")
            print(f"开始训练 Superclass Model (Classes: {num_super})...")
        super_model = train_linear_single_head(
            train_features, train_super_labels, super_map, num_super,
            feature_dim, batch_size, learning_rate, epochs, device, training_loss, verbose
        )
        if output_dir:
            torch.save(super_model.state_dict(), os.path.join(output_dir, "super_model.pth"))
            if verbose:
                print("超类模型已保存。")

        if verbose:
            print(f"开始训练 Subclass Model (Classes: {num_sub})...")
        sub_model = train_linear_single_head(
            train_features, train_sub_labels, sub_map, num_sub,
            feature_dim, batch_size, learning_rate, epochs, device, training_loss, verbose
        )
        if output_dir:
            torch.save(sub_model.state_dict(), os.path.join(output_dir, "sub_model.pth"))
            if verbose:
                print("子类模型已保存。")
    
    # 生成 super_to_sub 映射
    super_to_sub = create_super_to_sub_mapping(train_super_labels, train_sub_labels, output_dir, verbose)
    
    if output_dir and verbose:
        print("\n--- 所有模型训练完毕 ---")
        print(f"请检查 {output_dir} 目录下的模型文件 (.pth) 和 映射文件 (.json)。")
    
    # 返回结果
    if enable_feature_gating:
        return model, super_map, sub_map, super_to_sub
    else:
        return super_model, sub_model, super_map, sub_map, super_to_sub



