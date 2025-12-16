import torch
import numpy as np
import os

from src.augmentation_CAC_OpenMax.train_augmentated_CAC_OpenMax import train_augmentated_cac_openmax_classifier
from src.OpenMax.test_OpenMax import test_OpenMax_openset
from src.core.config import *
from src.core.training import create_label_mapping

# ================= 配置区域 =================
CONFIG = {
    "feature_dir": os.path.join(config.paths.split_features, "augmentation_features"),
    # "output_dir": os.path.join(config.paths.dev, "augmentation_CAC"),
    "output_dir": None,
    "model_name": "Augmentated_CAC_OpenMax",
    "feature_dim": config.model.feature_dim,
    "learning_rate": 0.01,
    "batch_size": config.experiment.batch_size,
    "epochs": 300,
    "novel_sub_index": 87,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # CAC
    "alpha_CAC": 10.0,
    "lambda_w": 0,
    "lambda_open": 0.3,
    # "anchor_mode": "uniform_hypersphere",
    "anchor_mode": "axis_aligned",
    "metric": "AUROC",
    "se_reduction": -1,
    # OpenMax
    "alpha_openmax": 5,
    "weibull_tail_size": 10,
    "distance_type": "euclidean",
    # "distance_type": "cosine",
}

SEEDS = [42, 123, 456, 789, 1024]  # 定义5个种子

# os.makedirs(CONFIG["output_dir"], exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print(f"CAC 多种子评估脚本 | Seeds: {SEEDS}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 加载数据
    # ---------------------------------------------------------
    print(">>>加载数据...")
    feature_dir = CONFIG["feature_dir"]

    # 训练集
    train_features = torch.load(os.path.join(feature_dir, "train_features.pt"))
    train_sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))

    # 验证集
    val_features = torch.load(os.path.join(feature_dir, "val_features.pt"))
    val_sub_labels = torch.load(os.path.join(feature_dir, "val_sub_labels.pt"))

    # 测试集
    test_features = torch.load(os.path.join(feature_dir, "test_features.pt"))
    test_sub_labels = torch.load(os.path.join(feature_dir, "test_sub_labels.pt"))

    # 映射
    num_classes, sub_map = create_label_mapping(train_sub_labels, "sub", CONFIG["output_dir"])

    NOVEL_ID = CONFIG["novel_sub_index"]
    print(NOVEL_ID)
    sub_map[NOVEL_ID] = -1

    # ---------------------------------------------------------
    # 运行多种子
    # ---------------------------------------------------------
    all_results = []

    for seed in SEEDS:
        print(f"\n" + "-" * 40)
        print(f"开始运行种子: {seed}")
        print("-" * 40)

        # === 训练 ===
        system = train_augmentated_cac_openmax_classifier(
            train_features=train_features,
            train_labels=train_sub_labels,
            val_features=val_features,
            val_labels=val_sub_labels,
            label_map=sub_map,
            num_classes=num_classes-1,
            model_name=CONFIG["model_name"],
            feature_dim=CONFIG["feature_dim"],
            batch_size=CONFIG["batch_size"],
            learning_rate=CONFIG["learning_rate"],
            epochs=CONFIG["epochs"],
            device=CONFIG["device"],
            output_dir=CONFIG["output_dir"],
            metric=CONFIG["metric"],
            seed=seed,
            # CAC
            anchor_mode=CONFIG["anchor_mode"],
            alpha_CAC=CONFIG["alpha_CAC"],
            lambda_w=CONFIG["lambda_w"],
            lambda_open=CONFIG["lambda_open"],
            se_reduction=CONFIG["se_reduction"],
            #OpenMax
            weibull_tail_size=CONFIG["weibull_tail_size"],
            alpha_openmax=CONFIG["alpha_openmax"],
            distance_type=CONFIG["distance_type"]
        )

        # === 测试 ===
        metrics = test_OpenMax_openset(
            system=system,
            test_features=test_features,
            test_labels=test_sub_labels,
            label_map=sub_map
        )

        all_results.append(metrics)
        print(f"    [Seed {seed} 结果] "
              f"Seen Acc: {metrics['acc_seen']:.2f}% | "
              f"Unseen Acc: {metrics['acc_unknown']:.2f}% | "
              f"AUROC: {metrics['auroc']:.4f}")

    # ---------------------------------------------------------
    # 统计与输出
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"最终评估报告 (平均值 ± 标准差, 运行 {len(SEEDS)} 次)")
    print("=" * 60)

    stat_keys = [
        ("acc_overall", "[Subclass] Overall"),
        ("acc_seen", "[Subclass] Seen"),
        ("acc_unknown", "[Subclass] Unseen"),
        ("auroc", "AUROC")
    ]

    for key, name in stat_keys:
        values = [res[key] for res in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)

        if key == "auroc":
            print(f"{name:30s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{name:30s}: {mean_val:.2f}% ± {std_val:.2f}%")

    print("=" * 60)
    print(f"结果已保存至: {CONFIG['output_dir']}")