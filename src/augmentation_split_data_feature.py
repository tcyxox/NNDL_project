import os
import numpy as np
import torch
from core.config import config
from core.utils import set_seed

CONFIG = {
    "feature_dir": os.path.join(config.paths.features, "augmentation_features"),
    "output_dir": os.path.join(config.paths.split_features, "augmentation_features"),

    "novel_ratio": config.split.novel_ratio,
    "train_ratio": config.split.train_ratio,
    "val_test_ratio": config.split.val_test_ratio,
    "novel_super_index": config.osr.novel_super_index,
    "novel_sub_index": config.osr.novel_sub_index,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def save_split(name, features, super_labels, sub_labels):
    """辅助函数：保存切分好的数据集"""
    torch.save(features, os.path.join(CONFIG["output_dir"], f"{name}_features.pt"))
    torch.save(super_labels, os.path.join(CONFIG["output_dir"], f"{name}_super_labels.pt"))
    torch.save(sub_labels, os.path.join(CONFIG["output_dir"], f"{name}_sub_labels.pt"))
    print(f"  [{name}] Saved: {features.shape[0]} samples")


if __name__ == "__main__":
    set_seed(config.experiment.seed)

    # ================= 1. 加载真实数据 (Real Data) =================
    print("正在加载全量真实特征...")
    features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))

    # ================= 2. 加载生成数据 (Generated Data) =================
    print("正在加载生成特征 (OpenMix/Generated)...")
    gen_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_openmix_features.pt"))
    num_gen = gen_features.shape[0]
    print(f"  -> 发现 {num_gen} 个生成样本")

    # 构造生成数据的标签：全部标记为 "未知类"
    gen_sub_labels = torch.full((num_gen,), CONFIG["novel_sub_index"], dtype=torch.long)
    gen_super_labels = torch.full((num_gen,), CONFIG["novel_super_index"], dtype=torch.long)

    # ================= 3. 划分已知/未知类逻辑 =================
    all_subclasses = torch.unique(sub_labels).numpy()
    np.random.shuffle(all_subclasses)

    num_novel = int(len(all_subclasses) * CONFIG["novel_ratio"])
    novel_classes = set(all_subclasses[:num_novel])  # 真实数据中的未知类
    known_classes = set(all_subclasses[num_novel:])  # 真实数据中的已知类

    print(f"\n=== 类别划分逻辑 ===")
    print(f"真实子类总数: {len(all_subclasses)}")
    print(f"设定为已知类 (Known): {len(known_classes)} 个")
    print(f"设定为未知类 (Novel): {len(novel_classes)} 个 (仅用于验证/测试)")

    # 构建索引掩码 (仅针对真实数据)
    is_known = torch.tensor([s.item() in known_classes for s in sub_labels])
    known_indices = torch.where(is_known)[0]
    novel_indices = torch.where(~is_known)[0]

    # ================= 4. 切分数据集索引 =================
    # 4.1 处理已知类 (Known): 分为 Train / Val / Test
    known_perm = torch.randperm(len(known_indices))
    known_indices = known_indices[known_perm]

    n_known = len(known_indices)
    n_train_known = int(n_known * CONFIG["train_ratio"])
    n_val_test_known = n_known - n_train_known
    n_val_known = int(n_val_test_known * CONFIG["val_test_ratio"])

    idx_train_known = known_indices[:n_train_known]
    idx_val_known = known_indices[n_train_known:n_train_known + n_val_known]
    idx_test_known = known_indices[n_train_known + n_val_known:]

    # 4.2 处理未知类 (Novel): 只分为 Val / Test
    novel_perm = torch.randperm(len(novel_indices))
    novel_indices = novel_indices[novel_perm]

    n_novel = len(novel_indices)
    n_val_novel = int(n_novel * CONFIG["val_test_ratio"])

    idx_val_novel = novel_indices[:n_val_novel]
    idx_test_novel = novel_indices[n_val_novel:]

    # ================= 5. 组合并保存 =================
    print(f"\n=== 数据集构建 ===")

    # --- 构建 Train Set ---
    # 逻辑：Train = [真实已知类] + [生成样本(伪未知类)]
    train_feat = torch.cat([features[idx_train_known], gen_features])
    train_super = torch.cat([super_labels[idx_train_known], gen_super_labels])
    train_sub = torch.cat([sub_labels[idx_train_known], gen_sub_labels])

    # 再次打乱 Train 顺序，让生成样本和真实样本混合
    perm_train = torch.randperm(len(train_feat))
    save_split("train",
               train_feat[perm_train],
               train_super[perm_train],
               train_sub[perm_train])

    # --- 构建 Val Set (纯净验证) ---
    # 逻辑：Val = [真实已知类] + [真实未知类] (不包含生成样本)
    val_feat = torch.cat([features[idx_val_known], features[idx_val_novel]])
    val_super = torch.cat([super_labels[idx_val_known], super_labels[idx_val_novel]])

    # 修改标签: 真实未知类也被标记为 novel_sub_index
    val_sub_known = sub_labels[idx_val_known]
    val_sub_novel = torch.full((len(idx_val_novel),), CONFIG["novel_sub_index"], dtype=torch.long)
    val_sub = torch.cat([val_sub_known, val_sub_novel])

    perm_val = torch.randperm(len(val_feat))
    save_split("val", val_feat[perm_val], val_super[perm_val], val_sub[perm_val])

    # --- 构建 Test Set (纯净测试) ---
    # 逻辑：Test = [真实已知类] + [真实未知类] (不包含生成样本)
    test_feat = torch.cat([features[idx_test_known], features[idx_test_novel]])
    test_super = torch.cat([super_labels[idx_test_known], super_labels[idx_test_novel]])

    test_sub_known = sub_labels[idx_test_known]
    test_sub_novel = torch.full((len(idx_test_novel),), CONFIG["novel_sub_index"], dtype=torch.long)
    test_sub = torch.cat([test_sub_known, test_sub_novel])

    perm_test = torch.randperm(len(test_feat))
    save_split("test", test_feat[perm_test], test_super[perm_test], test_sub[perm_test])

    print(f"\n完成！")
    print(f"Train集: 真实已知类 ({len(idx_train_known)}) + 生成未知类 ({len(gen_features)}) = {len(train_feat)} 样本")
    print(f"Val/Test集: 仅包含真实数据，未混入生成样本。")