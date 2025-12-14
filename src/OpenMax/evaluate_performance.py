import torch
import numpy as np
import os

from src.OpenMax.train_OpenMax import train_openmax_classifier
from src.OpenMax.test_OpenMax import test_OpenMax_openset
from src.core.config import *
from src.core.training import create_label_mapping

# ================= 配置区域 =================
CONFIG = {
    "feature_dir": config.paths.split_features,
    # "output_dir": os.path.join(config.paths.dev, "OpenMax"),
    "output_dir": None,
    "feature_dim": config.model.feature_dim,
    "learning_rate": 0.001,
    "batch_size": config.experiment.batch_size,
    "epochs": 100,
    "alpha": 2,
    "weibull_tail_size": 3,
    "distance_type": "cosine",
    "novel_sub_index": config.osr.novel_sub_index,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

SEEDS = [42, 123, 456, 789, 1024]

if CONFIG["output_dir"]:
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print(f"OpenMax 多种子评估脚本 | Seeds: {SEEDS}")
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
        system = train_openmax_classifier(
            train_features=train_features,
            train_labels=train_sub_labels,
            val_features=val_features,
            val_labels=val_sub_labels,
            label_map=sub_map,
            num_classes=num_classes,
            model_name="OpenMax_Model",
            feature_dim=CONFIG["feature_dim"],
            alpha=CONFIG["alpha"],
            weibull_tail_size=CONFIG["weibull_tail_size"],
            distance_type=CONFIG["distance_type"],
            output_dir=CONFIG["output_dir"],
            seed=seed,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["learning_rate"],
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