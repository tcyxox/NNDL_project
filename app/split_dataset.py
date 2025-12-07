import torch
import os
import numpy as np
from collections import Counter

# ================= 配置 =================
CONFIG = {
    "feature_dir": "precomputed_features",
    "output_dir": "split_data_osr",  # 新的数据存放位置
    "novel_ratio": 0.2,  # 20% 的子类将被作为“域外/未知”类保留
    "val_ratio": 0.1,   # 从已知类中划出 10% 做验证
    "test_ratio": 0.1,  # 从已知类中划出 10% 做测试

    # 标签设定
    "novel_sub_index": 87,   # 未知子类的 ID
    "novel_super_index": 3,  # 未知超类的 ID
    "seed": 42  # 固定随机种子，保证每次划分一致
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def save_split(name, features, super_labels, sub_labels):
    """辅助函数：保存切分好的数据集"""
    torch.save(features, os.path.join(CONFIG["output_dir"], f"{name}_features.pt"))
    torch.save(super_labels, os.path.join(CONFIG["output_dir"], f"{name}_super_labels.pt"))
    torch.save(sub_labels, os.path.join(CONFIG["output_dir"], f"{name}_sub_labels.pt"))
    print(f"  [{name}] Saved: {features.shape[0]} samples")


if __name__ == "__main__":
    # 1. 加载全量训练数据
    print("正在加载全量特征...")
    features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))

    total_samples = features.shape[0]
    all_subclasses = torch.unique(sub_labels).numpy()

    # 2. 划分 "已知类" (Domain In) 和 "未知类" (Domain Out)
    np.random.seed(CONFIG["seed"])
    np.random.shuffle(all_subclasses)

    num_novel = int(len(all_subclasses) * CONFIG["novel_ratio"])
    novel_classes = set(all_subclasses[:num_novel])  # 这些类不参与训练
    known_classes = set(all_subclasses[num_novel:])  # 这些类参与训练

    print(f"\n=== 类别划分逻辑 ===")
    print(f"总子类数: {len(all_subclasses)}")
    print(f"设定为已知类 (Known): {len(known_classes)} 个")
    print(f"设定为未知类 (Novel): {len(novel_classes)} 个 (示例: {list(novel_classes)[:5]}...)")

    # 3. 构建索引掩码
    is_known = torch.tensor([s.item() in known_classes for s in sub_labels])

    # 获取索引
    known_indices = torch.where(is_known)[0]
    novel_indices = torch.where(~is_known)[0]

    # 4. 切分数据集
    # 4.1 处理已知类 (Known): 分为 Train / Val / Test
    # 打乱已知类样本
    known_perm = torch.randperm(len(known_indices))
    known_indices = known_indices[known_perm]

    n_known = len(known_indices)
    n_val_known = int(n_known * CONFIG["val_ratio"])
    n_test_known = int(n_known * CONFIG["test_ratio"])
    n_train_known = n_known - n_val_known - n_test_known

    idx_train = known_indices[:n_train_known]
    idx_val_known = known_indices[n_train_known: n_train_known + n_val_known]
    idx_test_known = known_indices[n_train_known + n_val_known:]

    # 4.2 处理未知类 (Novel): 只分为 Val / Test (训练集不能看！)
    # 打乱未知类样本
    novel_perm = torch.randperm(len(novel_indices))
    novel_indices = novel_indices[novel_perm]

    n_novel = len(novel_indices)
    n_val_novel = int(n_novel * CONFIG["val_ratio"] / (CONFIG["val_ratio"] + CONFIG["test_ratio"]))

    idx_val_novel = novel_indices[:n_val_novel]
    idx_test_novel = novel_indices[n_val_novel:]

    # 5. 组合并保存
    print(f"\n=== 数据集构建 ===")

    # --- 构建 Train Set (纯净，只有已知类) ---
    save_split("train",
               features[idx_train],
               super_labels[idx_train],
               sub_labels[idx_train])

    # --- 构建 Val Set (混合，已知+未知) ---
    val_feat = torch.cat([features[idx_val_known], features[idx_val_novel]])
    val_super = torch.cat([super_labels[idx_val_known], super_labels[idx_val_novel]])
    # 修改标签: 来自 novel_indices 的样本，子类标签改为 87
    val_sub_known = sub_labels[idx_val_known]
    val_sub_novel = torch.full((len(idx_val_novel),), CONFIG["novel_sub_index"], dtype=torch.long)
    val_sub = torch.cat([val_sub_known, val_sub_novel])
    # 打乱顺序保存
    perm_val = torch.randperm(len(val_feat))
    save_split("val", val_feat[perm_val], val_super[perm_val], val_sub[perm_val])

    # --- 构建 Test Set (混合，已知+未知) ---
    test_feat = torch.cat([features[idx_test_known], features[idx_test_novel]])
    test_super = torch.cat([super_labels[idx_test_known], super_labels[idx_test_novel]])
    # 修改标签
    test_sub_known = sub_labels[idx_test_known]
    test_sub_novel = torch.full((len(idx_test_novel),), CONFIG["novel_sub_index"], dtype=torch.long)
    test_sub = torch.cat([test_sub_known, test_sub_novel])
    # 打乱顺序保存
    perm_test = torch.randperm(len(test_feat))
    save_split("test", test_feat[perm_test], test_super[perm_test], test_sub[perm_test])

    print(f"\n完成！数据已保存至 {CONFIG['output_dir']} 文件夹。")
    print(f"Train集: 仅包含 {len(known_classes)} 个已知子类。")
    print(f"Val/Test集: 包含已知子类 + {len(novel_classes)} 个被标记为 label={CONFIG['novel_sub_index']} 的未知子类。")
