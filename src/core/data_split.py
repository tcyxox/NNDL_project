"""
数据集划分工具模块

提供将特征数据划分为 train/val/test 的功能
"""
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SplitDataset:
    """划分后的数据集"""
    # 训练集（仅已知类）
    train_features: torch.Tensor
    train_super_labels: torch.Tensor
    train_sub_labels: torch.Tensor
    # 验证集（已知+未知）
    val_features: torch.Tensor
    val_super_labels: torch.Tensor
    val_sub_labels: torch.Tensor
    # 测试集（已知+未知）
    test_features: torch.Tensor
    test_super_labels: torch.Tensor
    test_sub_labels: torch.Tensor
    # 元信息
    known_classes: set
    novel_classes: set


def split_features(
        feature_dir: str,
        novel_ratio: float = 0.2,
        train_ratio: float = 0.7,
        val_test_ratio: float = 0.5,
        novel_sub_index: int = 87,
        output_dir: str = None,
        verbose: bool = True
) -> SplitDataset:
    """
    将特征数据划分为 train/val/test 集

    Args:
        feature_dir: 包含原始特征文件的目录
        novel_ratio: 设为 novel 的子类比例
        train_ratio: 已知类中用于训练的比例
        val_test_ratio: 剩余部分中用于验证的比例
        novel_sub_index: novel 子类的标签 (默认 87)
        output_dir: 可选，若提供则保存划分结果到此目录
        verbose: 是否打印详细信息

    Returns:
        SplitDataset: 包含所有划分数据的对象
    """
    # 1. 加载全量训练数据
    if verbose:
        print("正在加载全量特征...")
    features = torch.load(os.path.join(feature_dir, "train_features.pt"))
    super_labels = torch.load(os.path.join(feature_dir, "train_super_labels.pt"))
    sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))

    all_subclasses = torch.unique(sub_labels).numpy()

    # 2. 划分 "已知类" 和 "未知类"
    np.random.shuffle(all_subclasses)
    num_novel = int(len(all_subclasses) * novel_ratio)
    novel_classes = set(all_subclasses[:num_novel])
    known_classes = set(all_subclasses[num_novel:])

    if verbose:
        print(f"类别划分: 已知类 {len(known_classes)} 个, 未知类 {len(novel_classes)} 个")

    # 3. 构建索引掩码
    is_known = torch.tensor([s.item() in known_classes for s in sub_labels])
    known_indices = torch.where(is_known)[0]
    novel_indices = torch.where(~is_known)[0]

    # 4. 切分已知类: Train / Val / Test
    known_perm = torch.randperm(len(known_indices))
    known_indices = known_indices[known_perm]

    n_known = len(known_indices)
    n_train = int(n_known * train_ratio)
    n_val_test = n_known - n_train
    n_val_known = int(n_val_test * val_test_ratio)

    idx_train = known_indices[:n_train]
    idx_val_known = known_indices[n_train:n_train + n_val_known]
    idx_test_known = known_indices[n_train + n_val_known:]

    # 5. 切分未知类: Val / Test （训练集不能有！）
    novel_perm = torch.randperm(len(novel_indices))
    novel_indices = novel_indices[novel_perm]

    n_novel = len(novel_indices)
    n_val_novel = int(n_novel * val_test_ratio)

    idx_val_novel = novel_indices[:n_val_novel]
    idx_test_novel = novel_indices[n_val_novel:]

    # 6. 组装数据集
    # Train (纯已知类)
    train_feat = features[idx_train]
    train_super = super_labels[idx_train]
    train_sub = sub_labels[idx_train]

    # Val (已知 + 未知, novel 子类标签改为 novel_sub_index)
    val_feat = torch.cat([features[idx_val_known], features[idx_val_novel]])
    val_super = torch.cat([super_labels[idx_val_known], super_labels[idx_val_novel]])
    val_sub = torch.cat([
        sub_labels[idx_val_known],
        torch.full((len(idx_val_novel),), novel_sub_index, dtype=torch.long)
    ])
    perm_val = torch.randperm(len(val_feat))
    val_feat, val_super, val_sub = val_feat[perm_val], val_super[perm_val], val_sub[perm_val]

    # Test (已知 + 未知)
    test_feat = torch.cat([features[idx_test_known], features[idx_test_novel]])
    test_super = torch.cat([super_labels[idx_test_known], super_labels[idx_test_novel]])
    test_sub = torch.cat([
        sub_labels[idx_test_known],
        torch.full((len(idx_test_novel),), novel_sub_index, dtype=torch.long)
    ])
    perm_test = torch.randperm(len(test_feat))
    test_feat, test_super, test_sub = test_feat[perm_test], test_super[perm_test], test_sub[perm_test]

    if verbose:
        print(f"Train: {len(train_feat)}, Val: {len(val_feat)}, Test: {len(test_feat)}")

    # 7. 可选保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, feat, sup, sub in [
            ("train", train_feat, train_super, train_sub),
            ("val", val_feat, val_super, val_sub),
            ("test", test_feat, test_super, test_sub)
        ]:
            torch.save(feat, os.path.join(output_dir, f"{name}_features.pt"))
            torch.save(sup, os.path.join(output_dir, f"{name}_super_labels.pt"))
            torch.save(sub, os.path.join(output_dir, f"{name}_sub_labels.pt"))
        if verbose:
            print(f"数据已保存至 {output_dir}")

    return SplitDataset(
        train_features=train_feat, train_super_labels=train_super, train_sub_labels=train_sub,
        val_features=val_feat, val_super_labels=val_super, val_sub_labels=val_sub,
        test_features=test_feat, test_super_labels=test_super, test_sub_labels=test_sub,
        known_classes=known_classes, novel_classes=novel_classes
    )
