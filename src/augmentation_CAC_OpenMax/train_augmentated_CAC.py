import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
import os

from src.augmentation_CAC_OpenMax.augmentated_CAC import CACProjector, CACLoss
from src.core.config import *
from src.core.training import create_label_mapping
from src.core.utils import set_seed


CONFIG = {
    "feature_dir": os.path.join(config.paths.split_features, "augmentation_features"),
    # "output_dir": os.path.join(config.paths.dev, "augmentation_CAC"),
    "output_dir": None,
    "feature_dim": config.model.feature_dim,
    "learning_rate": 0.01,
    "batch_size": config.experiment.batch_size,
    "epochs": 300,
    "alpha": 10.0,
    "lambda_w": 0.1,
    "anchor_mode": "axis_aligned",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

if CONFIG["output_dir"] is not None:
    os.makedirs(CONFIG["output_dir"], exist_ok=True)


def train_augmentated_cac_classifier(
        train_features, train_labels,
        val_features, val_labels,
        label_map, num_classes, feature_dim,
        batch_size, learning_rate, epochs, device,
        seed=114, model_name="Augmentated_CAC",
        anchor_mode="axis_aligned", alpha=10.0, lambda_w=0.1, lambda_open=0.1, se_reduction=-1,
        output_dir=None, metric="AUROC"
):
    set_seed(seed)

    model = CACProjector(feature_dim, num_classes, alpha=alpha, se_reduction=se_reduction, anchor_mode=anchor_mode)
    model.to(device)

    # 训练集映射
    mapped_train_labels = torch.tensor([label_map[l.item()] for l in train_labels], dtype=torch.long)
    train_dataset = TensorDataset(train_features, mapped_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 验证集映射
    mapped_val_labels = torch.tensor([label_map[l.item()] for l in val_labels], dtype=torch.long)
    val_dataset = TensorDataset(val_features, mapped_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 损失与优化
    criterion = CACLoss(lambda_w=lambda_w, lambda_open=lambda_open).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # 学习率调整策略
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)

    print(f"\n开始训练 {model_name} - {anchor_mode} - alpha: {alpha} - lambda_w: {lambda_w} - se_reduction: {se_reduction}...")

    best_val_metric = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # =============== 训练阶段 ===============
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # 修改：现在需要传入logits用于生成样本的loss计算
            logits, distances = model(inputs)
            loss = criterion(logits, distances, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # scheduler.step()  # 更新学习率

        # =============== 验证阶段 ===============
        model.eval()
        val_loss = 0.0
        correct = 0
        total_seen = 0

        # 存储 AUROC 计算所需的数据
        all_scores = []
        all_binary_labels = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                logits, distances = model(inputs)

                mask = targets != -1

                # ====== 对所有类别计算 AUROC 分数 ======
                # CAC Score: gamma = d * (1 - softmin(d))
                softmin = F.softmax(-distances, dim=1)
                scores_map = distances * (1 - softmin)
                min_scores, _ = torch.min(scores_map, dim=1)
                all_scores.extend(min_scores.cpu().tolist())
                is_known = mask.cpu().int().tolist()
                all_binary_labels.extend(is_known)

                # ====== 对已知类计算 Loss 和 Accuracy ======
                if mask.sum() > 0:
                    # 筛选出已知类
                    known_targets = targets[mask]
                    known_distances = distances[mask]

                    # 修改：现在需要传入logits用于生成样本的loss计算
                    loss = criterion(logits, known_distances, known_targets)
                    val_loss += loss.item()

                    # 寻找距离最近的锚点
                    _, predicted = torch.min(known_distances, 1)
                    total_seen += known_targets.size(0)
                    correct += (predicted == known_targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * correct / total_seen if total_seen > 0 else 0.0

        val_auroc = 0.5
        if len(np.unique(all_binary_labels)) == 2:
            val_auroc = roc_auc_score(all_binary_labels, -np.array(all_scores))

        # --- 打印日志 & 保存最佳模型 ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Seen Acc: {val_acc:.2f}% | "
                  f"Val AUROC: {val_auroc:.4f}")
        if metric == "ACC":
            # 记录验证准确率最高的模型状态
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                # torch.save(best_model_state, os.path.join(output_dir, f"{model_name}_best.pth"))
        elif metric == "AUROC":
            # 记录AUROC最高的模型状态
            if val_auroc > best_val_metric:
                best_val_metric = val_auroc
                best_model_state = copy.deepcopy(model.state_dict())
                # torch.save(best_model_state, os.path.join(output_dir, f"{model_name}_best.pth"))

    print(f"训练结束。最佳验证集 {metric} : {best_val_metric:.2f}%")

    # 加载最佳权重并返回
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if output_dir:
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f"best_cac_model_seed_{seed}_alpha_{alpha}_{anchor_mode}.pth"))

        print(f"已经保存权重：{os.path.join(output_dir, f"best_cac_model_seed_{seed}_alpha_{alpha}_{anchor_mode}.pth")}")

    return model


if __name__ == "__main__":
    # =============================== 加载划分后的训练数据 ===============================
    print("加载训练集 (Train Split)...")
    train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))
    print("加载验证集 (Val Split)...")
    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_sub_labels.pt"))

    # =============================== 创建标签映射 ===============================
    num_super, super_map = create_label_mapping(train_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", CONFIG["output_dir"])
    sub_map[87] = -1

    # ================= 训练  =================
    best_model = train_augmentated_cac_classifier(
        train_features=train_features,
        train_labels=train_sub_labels,
        val_features=val_features,
        val_labels=val_sub_labels,
        label_map=sub_map,
        num_classes=num_sub,
        model_name="Augmentated_CAC",
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        device=CONFIG["device"],
        output_dir=None,
        alpha=CONFIG["alpha"],
        lambda_w=CONFIG["lambda_w"],
        seed=42
    )
