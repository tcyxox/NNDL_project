import pandas as pd
import numpy as np
import json
import os

from config import *

# ================= 配置 =================
CONFIG = {
    # 文件读取路径
    "train_csv": os.path.join(DATA_RAW_DIR, "train_data.csv"),
    "img_dir": os.path.join(DATA_RAW_DIR, "train_images.csv"),
    "super_map_csv": os.path.join(DATA_RAW_DIR, "superclass_mapping.csv"),
    "sub_map_csv": os.path.join(DATA_RAW_DIR, "subclass_mapping.csv"),

    # 输出路径
    "split_train_path": os.path.join(SPLIT_DIR, "osr_train_split.json"),
    "split_val_path": os.path.join(SPLIT_DIR, "osr_val_split.json"),
    "split_test_path": os.path.join(SPLIT_DIR, "osr_test_split.json"),
    "label_mapping_path": os.path.join(SPLIT_DIR, "osr_label_mapping.json"),

    # OSR 划分策略
    "novel_subclass_ratio": NOVEL_RATIO,                # 每个超类中隐藏 20% 的子类
    "train_ratio": TRAIN_RATIO,                         # 从已知类样本中划出 80% 做训练
    "val_ratio": (1-TRAIN_RATIO)*VAL_TEST_RATIO,        # 从已知类样本中划出 10% 做验证
    "test_ratio": (1-TRAIN_RATIO)*(1-VAL_TEST_RATIO),   # 从已知类样本中划出 10% 做测试

    # 随机种子
    "seed": SEED
}


def load_mappings(super_path, sub_path):
    """读取映射文件，返回已知类别的索引列表"""
    super_df = pd.read_csv(super_path)
    sub_df = pd.read_csv(sub_path)

    # 过滤掉 class == 'novel' 的行
    # 通常 novel 的 index 是最大的，但为了保险我们用字符串匹配
    known_supers = super_df[super_df['class'] != 'novel']['index'].tolist()
    novel_super_idx = super_df[super_df['class'] == 'novel']['index'].values[0]

    known_subs_all = sub_df[sub_df['class'] != 'novel']['index'].tolist()
    novel_sub_idx = sub_df[sub_df['class'] == 'novel']['index'].values[0]  # 应该是 87

    print(f"映射文件读取完毕:")
    print(f"  > 已知超类 ID: {known_supers}")
    print(f"  > 未知超类 ID: {novel_super_idx}")
    print(f"  > 未知子类 ID: {novel_sub_idx}")

    return known_supers, novel_super_idx, novel_sub_idx


def save_split_json(filename, image_list, super_list, sub_list):
    """
    格式: [{"image": "x.jpg", "super_label": 1, "sub_label": 5}, ...]
    """
    data = []
    for img, s_lbl, sub_lbl in zip(image_list, super_list, sub_list):
        data.append({
            "image": str(img),
            "super_label": int(s_lbl),
            "sub_label": int(sub_lbl)
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  > 已保存 {filename}: {len(data)} 个样本")


if __name__ == "__main__":
    df = pd.read_csv(CONFIG['train_csv'])
    known_super_ids, NOVEL_SUPER_ID, NOVEL_SUB_ID = load_mappings(CONFIG['super_map_csv'], CONFIG['sub_map_csv'])

    train_sub_ids = set()  # 最终用于训练的子类集合
    novel_sub_ids = set()  # 被隐藏的子类集合

    np.random.seed(CONFIG["seed"])

    print("\n=== 子类划分 (按超类分层) ===")
    for super_id in known_super_ids:
        # 找出当前超类下，train_data 中实际存在的所有子类
        subs_in_super = df[df['superclass_index'] == super_id]['subclass_index'].unique()

        # 随机打乱
        np.random.shuffle(subs_in_super)
        # 计算切分点
        n_novel = int(len(subs_in_super) * CONFIG["novel_subclass_ratio"])
        # 切分: 前 n_novel 个作为未知，剩下的作为已知
        curr_novel_subs = subs_in_super[:n_novel]
        curr_train_subs = subs_in_super[n_novel:]

        novel_sub_ids.update(curr_novel_subs)
        train_sub_ids.update(curr_train_subs)

        print(f"超类 {super_id}: 共 {len(subs_in_super)} 子类 -> 隐藏 {len(curr_novel_subs)} (Novel) / 保留 {len(curr_train_subs)} (Known)")

    print(f"\n总计: 训练用已知子类 {len(train_sub_ids)} 个, 验证用未知子类 {len(novel_sub_ids)} 个")

    # ======================================= 划分数据集（known / novel）样本 =======================================
    # 获取每一行的 subclass_index
    row_sub_labels = df['subclass_index'].values
    all_indices = np.arange(len(df))

    # A. 筛选出属于 "已知子类" 的样本
    mask_known = np.isin(row_sub_labels, list(train_sub_ids))
    idx_known_all = all_indices[mask_known]

    # B. 筛选出属于 "未知子类" 的样本
    mask_novel = np.isin(row_sub_labels, list(novel_sub_ids))
    idx_novel_all = all_indices[mask_novel]

    # ======================================= 构建 Train / Val / Test 集 =======================================
    # --- 处理已知类样本 ---
    np.random.shuffle(idx_known_all)
    n_total_known = len(idx_known_all)
    n_train_known = int(n_total_known * CONFIG["train_ratio"])
    n_val_known = int(n_total_known * CONFIG["val_ratio"])
    n_test_known = int(n_total_known * CONFIG["test_ratio"])

    idx_train = idx_known_all[:n_train_known]    # 训练集（已知）
    idx_val_known = idx_known_all[n_train_known:n_train_known+n_val_known]  # 验证集（已知）
    idx_test_known = idx_known_all[n_train_known+n_val_known:n_train_known+n_val_known+n_test_known]  # 测试集（已知）

    # --- 处理未知类样本 ---
    # 全部放进验证集 (Val Novel)，标签需修改为 87 (NOVEL_SUB_ID)
    n_total_novel = len(idx_novel_all)
    n_val_novel = int(n_total_novel * CONFIG["val_ratio"])
    n_test_novel = int(n_total_novel * CONFIG["test_ratio"])

    idx_val_novel = idx_novel_all[:n_val_novel]
    idx_test_novel = idx_novel_all[n_val_novel:n_val_novel+n_test_novel]

    print(f"\n=== 数据集样本分布 ===")
    print(f"Train Set: {len(idx_train)} (纯已知类)")
    print(f"Val Set:   {len(idx_val_known) + len(idx_val_novel)} (含 {len(idx_val_known)} 已知 + {len(idx_val_novel)} 未知)")
    print(f"Test Set:  {len(idx_test_known) + len(idx_test_novel)} (含 {len(idx_test_known)} 已知 + {len(idx_test_novel)} 未知)")

    # ======================================= 生成 JSON 文件 =======================================
    # ----- train -----
    train_imgs = df.iloc[idx_train]['image'].tolist()
    train_super_lbls = df.iloc[idx_train]['superclass_index'].tolist()
    train_sub_lbls = df.iloc[idx_train]['subclass_index'].tolist()

    save_split_json(CONFIG["split_train_path"], train_imgs, train_super_lbls, train_sub_lbls)

    # ----- val -----
    # known (保持原标签)
    val_k_imgs = df.iloc[idx_val_known]['image'].tolist()
    val_k_super_lbls = df.iloc[idx_val_known]['superclass_index'].tolist()
    val_k_sub_lbls = df.iloc[idx_val_known]['subclass_index'].tolist()

    # novel (标签改为 87/Novel)
    val_n_imgs = df.iloc[idx_val_novel]['image'].tolist()
    val_n_super_lbls = df.iloc[idx_val_novel]['superclass_index'].tolist()
    val_n_sub_lbls = [int(NOVEL_SUB_ID)] * len(val_n_imgs)

    # 合并
    val_imgs = val_k_imgs + val_n_imgs
    val_super_lbls = val_k_super_lbls + val_n_super_lbls
    val_sub_lbls = val_k_sub_lbls + val_n_sub_lbls

    # 打乱验证集顺序
    val_combined = list(zip(val_imgs, val_super_lbls, val_sub_lbls))
    np.random.shuffle(val_combined)
    val_imgs, val_super_lbls, val_sub_lbls = zip(*val_combined)

    save_split_json(CONFIG["split_val_path"], val_imgs, val_super_lbls, val_sub_lbls)

    # ----- test -----
    # known (保持原标签)
    test_k_imgs = df.iloc[idx_test_known]['image'].tolist()
    test_k_super_lbls = df.iloc[idx_test_known]['superclass_index'].tolist()
    test_k_sub_lbls = df.iloc[idx_test_known]['subclass_index'].tolist()

    # novel (标签改为 87/Novel)
    test_n_imgs = df.iloc[idx_test_novel]['image'].tolist()
    test_n_super_lbls = df.iloc[idx_test_novel]['superclass_index'].tolist()
    test_n_sub_lbls = [int(NOVEL_SUB_ID)] * len(test_n_imgs)

    # 合并
    test_imgs = test_k_imgs + test_n_imgs
    test_super_lbls = test_k_super_lbls + test_n_super_lbls
    test_sub_lbls = test_k_sub_lbls + test_n_sub_lbls

    # 打乱验证集顺序
    test_combined = list(zip(test_imgs, test_super_lbls, test_sub_lbls))
    np.random.shuffle(test_combined)
    test_imgs, test_super_lbls, test_sub_lbls = zip(*test_combined)

    save_split_json(CONFIG["split_test_path"], test_imgs, test_super_lbls, test_sub_lbls)

    # ----- 生成映射文件 (osr_label_mapping.json) -----
    # dict: {原始ID: 训练ID}
    sorted_train_ids = sorted(list(train_sub_ids))
    label_map = {int(original_id): int(new_id) for new_id, original_id in enumerate(sorted_train_ids)}

    with open(CONFIG["label_mapping_path"], "w") as f:
        json.dump(label_map, f)
    print(f"标签映射表已保存 (共 {len(label_map)} 类)")