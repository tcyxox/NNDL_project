import torch
import os

from core.config import config
from core.train import set_seed, create_label_mapping, train_classifier, train_hierarchical, create_super_to_sub_mapping
from core.models import HierarchicalClassifier

CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    "enable_feature_gating": config.experiment.enable_feature_gating
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(config.experiment.seed)


if __name__ == "__main__":
    # =============================== 加载划分后的训练数据 ===============================
    print("正在加载划分后的训练数据 (Train Split)...")
    train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))

    # =============================== 创建标签映射 ===============================
    num_super, super_map = create_label_mapping(train_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", CONFIG["output_dir"])

    # =============================== 训练模型 ===============================
    if CONFIG["enable_feature_gating"]:
        # 使用 HierarchicalClassifier 联合训练
        print("\n=== 使用 Soft Attention (联合训练模式) ===")
        model = HierarchicalClassifier(
            CONFIG["feature_dim"], num_super, num_sub
        )
        model = train_hierarchical(
            model, train_features, train_super_labels, train_sub_labels,
            super_map, sub_map, CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "hierarchical_model.pth"))
        print("层次模型已保存。")
    else:
        # 独立训练 (现有逻辑)
        print("\n=== 独立训练模式 ===")
        # 1. 训练超类模型
        super_model = train_classifier(
            train_features, train_super_labels, super_map, num_super, "Superclass Model",
            CONFIG["feature_dim"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        torch.save(super_model.state_dict(), os.path.join(CONFIG["output_dir"], "super_model.pth"))
        print("超类模型已保存。")

        # 2. 训练子类模型
        sub_model = train_classifier(
            train_features, train_sub_labels, sub_map, num_sub, "Subclass Model",
            CONFIG["feature_dim"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        torch.save(sub_model.state_dict(), os.path.join(CONFIG["output_dir"], "sub_model.pth"))
        print("子类模型已保存。")

    # 3. 生成超类到子类的映射表 (两种模式都需要)
    create_super_to_sub_mapping(train_super_labels, train_sub_labels, CONFIG["output_dir"])

    print("\n--- 所有模型训练完毕 ---")
    print(f"请检查 {CONFIG['output_dir']} 目录下的模型文件 (.pth) 和 映射文件 (.json)。")
