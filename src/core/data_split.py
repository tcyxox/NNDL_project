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
    # 验证集（根据模式：仅已知类 或 已知+未知）
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
        novel_subclass_ratio: float,
        train_ratio: float,
        val_test_ratio: float,
        val_include_novel: bool,
        novel_sub_index: int,
        novel_super_index: int,
        verbose: bool,
        force_super_novel: bool = False,
        output_dir: str = None
) -> SplitDataset:
    """
    将特征数据划分为 train/val/test 集

    Args:
        feature_dir: 包含原始特征文件的目录
        novel_subclass_ratio: 每个包含 novel class 的划分的 novel subclass 比例
        train_ratio: 已知类中用于训练的比例
        val_test_ratio: 剩余部分中用于验证的比例
        val_include_novel: 是否仅 test 含未知类（True: train/val 纯已知; False: val/test 都含未知）
        novel_sub_index: novel 子类的标签
        novel_super_index: novel 超类的标签
        verbose: 是否打印详细信息
        force_super_novel: 是否强制将一个超类设为 Novel 并分配给 Test (仅用于 submit)
        output_dir: 可选，若提供则保存划分结果到此目录

    Returns:
        SplitDataset: 包含所有划分数据的对象
    """
    # 1. 加载全量训练数据
    if verbose:
        print("正在加载全量特征...")
    features = torch.load(os.path.join(feature_dir, "train_features.pt"))
    super_labels = torch.load(os.path.join(feature_dir, "train_super_labels.pt"))
    sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))

    # 构建 sub -> super 映射
    unique_subs = torch.unique(sub_labels)
    sub_to_super = {}
    for sub in unique_subs:
        mask = sub_labels == sub
        super_cls = super_labels[mask][0].item()
        sub_to_super[sub.item()] = super_cls

    all_subclasses = unique_subs.numpy()
    all_superclasses = torch.unique(super_labels).numpy()

    # 2. 划分 "已知类" 和 "未知类"
    
    # 初始化变量
    novel_super_subclasses = set()
    novel_super_cls = -1
    
    # 只有当 force_super_novel=True 时，才强制选择一个 Super Novel Class
    if force_super_novel:
        # 策略：选择 ID 最大的那个超类作为 Novel Superclass (或者随机选)
        novel_super_cls = np.random.choice(all_superclasses)
        
        # 找出属于该超类的所有子类
        novel_super_subclasses = [sub for sub, sup in sub_to_super.items() if sup == novel_super_cls]
        novel_super_subclasses = set(novel_super_subclasses)
        
        if verbose:
            print(f"强制选定 Super Novel Class: {novel_super_cls} (包含 {len(novel_super_subclasses)} 个子类)")

    # 首先，Novel Superclass 的所有子类必须是 Novel (如果 force_super_novel=True)
    remaining_subclasses = [s for s in all_subclasses if s not in novel_super_subclasses]
    np.random.shuffle(remaining_subclasses)
    
    # 计算还需要多少个 Novel 子类
    # novel_ratio 表示每个 split 的未知类比例
    if val_include_novel:
        total_novel_ratio = novel_subclass_ratio
    else:
        total_novel_ratio = 2 * novel_subclass_ratio
    
    num_total_novel = int(len(all_subclasses) * total_novel_ratio)
    num_additional_novel = max(0, num_total_novel - len(novel_super_subclasses))
    
    additional_novel_classes = set(remaining_subclasses[:num_additional_novel])
    
    novel_classes = novel_super_subclasses.union(additional_novel_classes)
    known_classes = set(remaining_subclasses[num_additional_novel:])

    if verbose:
        if force_super_novel:
            print(f"类别划分: 已知类 {len(known_classes)} 个, "
                  f"未知类 {len(novel_classes)} 个 "
                  f"(其中 Super Novel 子类 {len(novel_super_subclasses)} 个, "
                  f"普通 Novel 子类 {len(additional_novel_classes)} 个)")
        else:
             print(f"类别划分: 已知类 {len(known_classes)} 个, 未知类 {len(novel_classes)} 个 (总比例 {total_novel_ratio*100:.0f}%)")

    # 4. 构建索引掩码
    is_known = torch.tensor([s.item() in known_classes for s in sub_labels])
    known_indices = torch.where(is_known)[0]
    
    # 将 Novel 样本分为两类：Super Novel 和 Ordinary Novel
    is_super_novel = torch.tensor([s.item() in novel_super_subclasses for s in sub_labels])
    is_ordinary_novel = torch.tensor([s.item() in additional_novel_classes for s in sub_labels])
    
    super_novel_indices = torch.where(is_super_novel)[0]
    ordinary_novel_indices = torch.where(is_ordinary_novel)[0]

    # 5. 切分已知类: Train / Val / Test
    known_perm = torch.randperm(len(known_indices))
    known_indices = known_indices[known_perm]

    n_known = len(known_indices)
    n_train = int(n_known * train_ratio)
    n_val_test = n_known - n_train
    n_val_known = int(n_val_test * val_test_ratio)

    idx_train = known_indices[:n_train]
    idx_val_known = known_indices[n_train:n_train + n_val_known]
    idx_test_known = known_indices[n_train + n_val_known:]

    # 6. 切分未知类
    # 策略：
    # 1. Super Novel 样本 -> 优先放入 Test (为了测试 Superclass OOD)
    # 2. Ordinary Novel 样本 -> 根据配置放入 Test 或 Val+Test
    
    if val_include_novel:
        # 模式A: 仅 Test 含未知类
        # 这种情况下，Val 必须纯已知，所以 Super Novel 只能去 Test
        idx_val_novel = torch.tensor([], dtype=torch.long)
        # 所有未知类（Super + Ordinary）都给 Test
        idx_test_novel = torch.cat([super_novel_indices, ordinary_novel_indices])
    else:
        # 模式B: Val 和 Test 都含未知类
        
        # 处理 Ordinary Novel
        ord_novel_subclasses = list(additional_novel_classes)
        np.random.shuffle(ord_novel_subclasses)
        
        if not force_super_novel:
            # 原始逻辑：根据 val_test_ratio 划分未知类
            # 此时 super_novel_indices 为空，只有 ordinary_novel_indices (即所有 novel)
            n_val_novel_classes = int(len(ord_novel_subclasses) * val_test_ratio)
            val_ord_novel_classes = set(ord_novel_subclasses[:n_val_novel_classes])
            test_ord_novel_classes = set(ord_novel_subclasses[n_val_novel_classes:])
            
            # 构建索引
            is_val_ord_novel = torch.tensor([s.item() in val_ord_novel_classes for s in sub_labels])
            idx_val_novel = torch.where(is_val_ord_novel)[0]
            
            is_test_ord_novel = torch.tensor([s.item() in test_ord_novel_classes for s in sub_labels])
            idx_test_novel = torch.where(is_test_ord_novel)[0]
            
        else:
            # 强制 Super Novel 逻辑：Super Novel 尽量给 Val
            # 计算 Test 需要分配多少 novel 类
            n_test_novel_target = int(len(all_subclasses) * novel_subclass_ratio)
            
            # Super Novel 全给 Val，所以 Test 从 Ordinary Novel 中取
            n_test_from_ord = min(len(ord_novel_subclasses), n_test_novel_target)
            
            test_ord_novel_classes = set(ord_novel_subclasses[:n_test_from_ord])
            val_ord_novel_classes = set(ord_novel_subclasses[n_test_from_ord:])
            
            # 构建索引
            # Test 只包含 Ordinary Novel
            is_test_ord_novel = torch.tensor([s.item() in test_ord_novel_classes for s in sub_labels])
            idx_test_novel = torch.where(is_test_ord_novel)[0]
            
            # Val 包含所有 Super Novel 和剩余的 Ordinary Novel
            is_val_ord_novel = torch.tensor([s.item() in val_ord_novel_classes for s in sub_labels])
            idx_val_ord_novel = torch.where(is_val_ord_novel)[0]
            idx_val_novel = torch.cat([super_novel_indices, idx_val_ord_novel])
        
    if verbose:
        print(f"未知样本分配: Val {len(idx_val_novel)}, Test {len(idx_test_novel)} "
              f"(含 Super Novel {len(super_novel_indices)})")

    # 7. 组装数据集
    # 辅助函数：处理标签
    def process_labels(indices, is_novel_super_override=False):
        feat = features[indices]
        sup = super_labels[indices].clone() # Clone to avoid modifying original
        sub = sub_labels[indices].clone()
        
        # 如果样本属于 Super Novel Class，将其 Super Label 设为 novel_super_index
        # 注意：这里我们已经根据 indices 选好了样本。
        # 上面逻辑保证了 super_novel_indices 里的样本确实属于那个被选中的 novel superclass。
        # 我们需要把这些样本的 super label 改成 novel_super_index (比如 3)
        # 而对于 ordinary novel (属于已知 superclass 但未知 subclass)，super label 保持不变
        
        # 方法：检查每个样本的原始 super label 是否等于 novel_super_cls
        # 如果是，则修改为 novel_super_index
        # 或者更简单：我们知道 super_novel_indices 里的样本都是 novel_super_cls
        # 但传入的是一般 indices，混合了各种情况
        
        # 在 idx 对应的样本中，找到属于 novel_super_cls 的，修改标签
        mask_is_super_novel = (sup == novel_super_cls)
        sup[mask_is_super_novel] = novel_super_index
        
        # 处理 Sub Labels: 所有 Novel 的 Sub Label 设为 novel_sub_index
        # 判断是否是 Novel Class (Super or Ordinary)
        # 我们可以利用 known_classes 集合
        is_novel_sub = torch.tensor([s.item() in novel_classes for s in sub])
        sub[is_novel_sub] = novel_sub_index
        
        return feat, sup, sub

    # Train (纯已知类)
    train_feat, train_super, train_sub = process_labels(idx_train)

    # Val
    if len(idx_val_novel) > 0:
        val_indices = torch.cat([idx_val_known, idx_val_novel])
    else:
        val_indices = idx_val_known
    val_feat, val_super, val_sub = process_labels(val_indices)
    
    perm_val = torch.randperm(len(val_feat))
    val_feat, val_super, val_sub = val_feat[perm_val], val_super[perm_val], val_sub[perm_val]

    # Test
    test_indices = torch.cat([idx_test_known, idx_test_novel])
    test_feat, test_super, test_sub = process_labels(test_indices)
    
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

