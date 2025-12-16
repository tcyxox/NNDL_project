"""
统一的提交脚本 - 合并训练、调参和推理流程

运行: python submit.py

流程:
  Step 1: 划分数据 (训练集 + 验证集)
  Step 2: 用训练集训练模型
  Step 3: 在验证集上计算阈值
  Step 4: 在测试集上推理
  Step 5: 生成提交 CSV
"""
import os

import pandas as pd
import torch
import numpy as np

from src.core.config import config
from src.core.data_split import split_features
from src.core.utils import set_seed
from src.core.training import create_label_mapping

from src.CAC_OpenMax.train_CAC_OpenMax import train_cac_openmax_classifier


# ===================== 配置 =====================
CONFIG = {
    # 数据路径
    "feature_dir": config.paths.features,
    "output_csv": os.path.join(config.paths.outputs, "submission_osr.csv"),

    # 数据划分
    "novel_ratio": config.split.novel_ratio,
    "train_ratio": config.split.train_ratio,
    "val_test_ratio": config.split.val_test_ratio,

    # OSR 标签
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,

    # ============================================================
    "model_name": "CAC_OpenMax",
    # "output_dir": os.path.join(config.paths.dev, "CAC_OpenMax"),
    "output_dir": None,
    "feature_dim": config.model.feature_dim,
    # "feature_dim": 768,
    "learning_rate": 0.01,
    "batch_size": config.experiment.batch_size,
    "epochs": 300,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "novel_sub_index": config.osr.novel_sub_index,
    # CAC
    "alpha_CAC": 10.0,
    "lambda_w": 0.1,
    # "anchor_mode": "uniform_hypersphere",
    "anchor_mode": "axis_aligned",
    "metric": "Last",
    "se_reduction": -1,
    # OpenMax
    "alpha_openmax": 3,
    "weibull_tail_size": 5,
    "distance_type": "euclidean",
    # "distance_type": "cosine",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    set_seed(config.experiment.seed)

    print("=" * 70)
    print("Unified Submission Pipeline")
    print("=" * 70)

    # ======================================= Step 1: 划分数据 =======================================
    print("\n--- Step 1: 划分数据 ---")

    data = split_features(
        feature_dir=CONFIG["feature_dir"],
        novel_ratio=0,
        train_ratio=1,
        val_test_ratio=0,
        novel_sub_index=CONFIG["novel_sub_idx"],
        output_dir=None,  # 不保存中间文件
        verbose=True
    )

    # 合并 train + val + test 作为完整训练数据
    # (因为提交时我们要用尽可能多的数据训练)
    full_train_features = torch.cat([data.train_features, data.val_features, data.test_features], dim=0)
    full_train_super = torch.cat([data.train_super_labels, data.val_super_labels, data.test_super_labels], dim=0)
    full_train_sub = torch.cat([data.train_sub_labels, data.val_sub_labels, data.test_sub_labels], dim=0)

    print(full_train_features.shape)

    sub_num_classes, sub_map = create_label_mapping(full_train_sub, "sub", CONFIG["output_dir"])
    super_num_classes, super_map = create_label_mapping(full_train_super, "sub", CONFIG["output_dir"])

    print(f"  > 完整训练样本数: {len(full_train_features)} (训练{len(data.train_features)} + 验证{len(data.val_features)} + 测试{len(data.test_features)})")
    print(f"  > 含有已知超类共: {super_num_classes}")
    # print(full_train_super.unique())
    print(f"  > 含有已知子类共: {sub_num_classes}")
    # print(full_train_sub.unique())

    # ======================================= Step 2: 训练模型 =======================================
    print("\n--- Step 2.1: 训练子类模型 ---")
    system_sub = train_cac_openmax_classifier(
        train_features=full_train_features,
        train_labels=full_train_sub,
        val_features=full_train_features,
        val_labels=full_train_sub,
        label_map=sub_map,
        num_classes=sub_num_classes,
        model_name=CONFIG["model_name"],
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        device=CONFIG["device"],
        output_dir=CONFIG["output_dir"],
        metric=CONFIG["metric"],
        seed=0,
        # CAC
        anchor_mode=CONFIG["anchor_mode"],
        alpha_CAC=CONFIG["alpha_CAC"],
        lambda_w=CONFIG["lambda_w"],
        se_reduction=CONFIG["se_reduction"],
        # OpenMax
        weibull_tail_size=CONFIG["weibull_tail_size"],
        alpha_openmax=CONFIG["alpha_openmax"],
        distance_type=CONFIG["distance_type"]
    )

    print("\n--- Step 2.2: 训练超类模型 ---")
    system_super = train_cac_openmax_classifier(
        train_features=full_train_features,
        train_labels=full_train_super,
        val_features=full_train_features,
        val_labels=full_train_super,
        label_map=super_map,
        num_classes=super_num_classes,
        model_name=CONFIG["model_name"],
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"] - 100,
        device=CONFIG["device"],
        output_dir=CONFIG["output_dir"],
        metric=CONFIG["metric"],
        seed=0,
        # CAC
        anchor_mode=CONFIG["anchor_mode"],
        alpha_CAC=CONFIG["alpha_CAC"],
        lambda_w=CONFIG["lambda_w"],
        se_reduction=CONFIG["se_reduction"],
        # OpenMax
        weibull_tail_size=CONFIG["weibull_tail_size"],
        alpha_openmax=CONFIG["alpha_openmax"],
        distance_type=CONFIG["distance_type"]
    )

    # ======================================= Step 4: 测试集推理 =======================================
    print("\n--- Step 4: 测试集推理 ---")

    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_image_names = torch.load(os.path.join(CONFIG["feature_dir"], "test_image_names.pt"))
    print(f"  > 真实测试样本数: {len(test_features)}")

    # 获取预测结果
    # system.predict 返回的类别中: 0 是未知, 1 是原类别0, 2 是原类别1 ...
    # sub class
    sub_pred_indices, _ = system_sub.predict(test_features)
    sub_pred_indices = np.where(sub_pred_indices == 0, 87, sub_pred_indices-1) # 重新对齐 87为新类别
    # super class
    super_pred_indices, _ = system_super.predict(test_features)
    super_pred_indices = np.where(super_pred_indices == 0, 3, super_pred_indices - 1)  # 重新对齐 3为新类别

    # === Step 5: 生成提交文件 ===
    print("\n--- Step 5: 生成提交文件 ---")

    predictions = [
        {"image": test_image_names[i], "superclass_index": super_pred_indices[i], "subclass_index": sub_pred_indices[i]}
        for i in range(len(test_image_names))
    ]

    df = pd.DataFrame(predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"  > 提交文件已保存至: {CONFIG['output_csv']}")
    print(f"  > 共 {len(predictions)} 条预测")

    # 统计信息
    novel_super_count = sum(1 for p in super_pred_indices if p == CONFIG["novel_super_idx"])
    novel_sub_count = sum(1 for p in sub_pred_indices if p == CONFIG["novel_sub_idx"])
    print(f"  > Novel superclass 预测数: {novel_super_count}")
    print(f"  > Novel subclass 预测数: {novel_sub_count}")

    print("\n" + "=" * 70)
    print("提交完成!")
    print("=" * 70)

