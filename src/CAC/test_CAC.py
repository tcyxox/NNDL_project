from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os

from src.core.config import *


CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "batch_size": config.experiment.batch_size,
    "target_recall": config.experiment.target_recall,
    "alpha": 10.0,
    "lambda_w": 0.1,
    "anchor_mode": "axis_aligned",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def calculate_metrics(y_true, y_pred, novel_label):
    """计算准确率指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc_all = accuracy_score(y_true, y_pred)

    mask_known = (y_true != novel_label)
    acc_known = accuracy_score(y_true[mask_known], y_pred[mask_known]) if np.sum(mask_known) > 0 else 0.0

    mask_novel = (y_true == novel_label)
    acc_novel = accuracy_score(y_true[mask_novel], y_pred[mask_novel]) if np.sum(mask_novel) > 0 else 0.0

    return acc_all, acc_known, acc_novel


def get_cac_scores(model, loader, device):
    """辅助函数：计算 CAC 距离分数"""
    scores = []
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            _, distances = model(inputs)

            # CAC Score 计算: Distance * (1 - Softmin)
            # 越小越代表是已知类
            softmin = F.softmax(-distances, dim=1)
            rejection_scores_vector = distances * (1 - softmin)
            min_scores, predicted = torch.min(rejection_scores_vector, dim=1)

            scores.extend(min_scores.cpu().tolist())
            preds.extend(predicted.cpu().tolist())
            targets.extend(target.tolist())

    return np.array(scores), np.array(preds), np.array(targets)


def test_cac_openset(model,
                     test_features, test_labels,
                     val_features, val_labels,
                     label_map, device, target_recall, novel_label=-1):
    """
    Returns:
        dict: 包含 acc_seen (已知类精度), acc_unknown (未知类精度), auroc 的字典
    """
    model.eval()
    model.to(device)

    # 过滤未知样本
    mapped_val_labels = torch.tensor([label_map[l.item()] for l in val_labels], dtype=torch.long)
    mask_known = mapped_val_labels != -1
    mapped_val_labels = mapped_val_labels[mask_known]
    val_features = val_features[mask_known]
    val_dataset = TensorDataset(val_features, mapped_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    mapped_test_labels = torch.tensor([label_map[l.item()] for l in test_labels], dtype=torch.long)
    test_dataset = TensorDataset(test_features, mapped_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ======================================  计算阈值 ======================================
    # 获取验证集分数
    val_scores, _, _ = get_cac_scores(model, val_loader, device)
    # 计算阈值
    threshold = np.percentile(val_scores, target_recall * 100)
    print(f"  -> 计算得到的阈值 (Recall {target_recall}): {threshold:.4f}")

    # ====================================== 评估测试集 ======================================
    # 获取测试集所有分数和预测
    test_scores, raw_preds, test_targets = get_cac_scores(model, test_loader, device)

    # ====================================== 生成最终预测 ======================================
    final_preds = raw_preds.copy()
    reject_mask = test_scores > threshold
    final_preds[reject_mask] = novel_label  # 拒绝样本标记为 -1

    # ====================================== 计算指标 ======================================
    acc_all, acc_known, acc_novel = calculate_metrics(test_targets, final_preds, novel_label=novel_label)

    # AUROC (整体区分度，与阈值无关) ---
    auroc = 0.5
    known_mask = (test_targets != novel_label)
    unknown_mask = (test_targets == novel_label)

    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        binary_targets = np.zeros_like(test_targets)
        binary_targets[known_mask] = 1

        # roc_auc_score 默认 score 越大越是 Positive(1/Known)
        # CAC 距离越小越是 Known，所以取负号
        auroc = roc_auc_score(binary_targets, -test_scores)

    return {
        "acc_overall": acc_all * 100,
        "acc_seen": acc_known * 100,
        "acc_unknown": acc_novel * 100,
        "auroc": auroc,
        "threshold": threshold
    }


if __name__ == "__main__":
    from src.core.training import create_label_mapping
    from src.CAC.CAC import CACProjector
    print("加载测试集 (Test Split)...")
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt"))
    test_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_super_labels.pt"))
    test_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_sub_labels.pt"))

    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_sub_labels.pt"))

    num_super, super_map = create_label_mapping(test_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(test_sub_labels, "sub", CONFIG["output_dir"])
    sub_map[87] = -1

    model = CACProjector(CONFIG["feature_dim"], num_sub-1, alpha=CONFIG["alpha"], anchor_mode=CONFIG["anchor_mode"])
    model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], f"best_cac_model_seed_{42}.pth")))

    # ================= 测试 =================
    res = test_cac_openset(
        model=model,
        test_features=test_features,
        test_labels=test_sub_labels,
        val_features=val_features,
        val_labels=val_sub_labels,
        label_map=sub_map,
        target_recall=CONFIG["target_recall"],
        device=CONFIG["device"]
    )

    print(res)
