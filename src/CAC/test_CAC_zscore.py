from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os

from src.core.config import *
from src.core.training import create_label_mapping
from src.CAC.CAC import *


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

    # --- Energy & Z-Score 参数 ---
    "std_multiplier": 0.5,   # Z-Score 的 k 值 (Mean - k * Std)
    "temperature": 1.0,      # Energy Score 的温度参数
    # ---------------------------
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


def compute_energy_score(logits, temperature=1.0):
    """
    计算 Energy Score (LogSumExp)
    分数越高 -> 越可能是已知类
    """
    if temperature == 0:
        return torch.max(logits, dim=1)[0]
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def get_energy_data(model, loader, device, temperature=1.0):
    """
    辅助函数：获取 Logits, Energy Scores 和 基础预测
    """
    scores = []
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            # 获取模型输出
            # 注意：CACProjector 通常返回 (logits, distances)
            # 这里我们需要 logits
            logits, _ = model(inputs)

            # 1. 计算 Energy Score
            batch_scores = compute_energy_score(logits, temperature)

            # 2. 获取基础分类预测 (Argmax Logits)
            _, batch_preds = torch.max(logits, dim=1)

            scores.extend(batch_scores.cpu().tolist())
            preds.extend(batch_preds.cpu().tolist())
            targets.extend(target.tolist())

    return np.array(scores), np.array(preds), np.array(targets)


def test_cac_openset(model,
                       test_features, test_labels,
                       val_features, val_labels,
                       label_map, device,
                       std_multiplier=3.0,
                       temperature=1.0,
                       novel_label=-1):
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
    val_scores, _, _ = get_energy_data(model, val_loader, device, temperature)
    # 计算阈值
    # 计算均值和标准差
    mean_score = np.mean(val_scores)
    std_score = np.std(val_scores)

    # 设定阈值: Mean - k * Std
    # 逻辑: 只有分数极低(偏离均值 k 个标准差)的才被认为是未知
    threshold = mean_score - std_multiplier * std_score
    print(f"  -> Val Mean: {mean_score:.4f}, Val Std: {std_score:.4f}")
    print(f"  -> Calculated Threshold (Mean - {std_multiplier}*Std): {threshold:.4f}")

    # ====================================== 评估测试集 ======================================
    # 获取测试集所有分数和预测
    test_scores, raw_preds, test_targets = get_energy_data(model, test_loader, device, temperature)

    # ====================================== 生成最终预测 ======================================
    final_preds = raw_preds.copy()
    # Energy Logic: 分数越高越确信。
    # 如果 Score < Threshold，则拒绝 (视为未知类)
    reject_mask = test_scores < threshold
    final_preds[reject_mask] = novel_label

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
        auroc = roc_auc_score(binary_targets, test_scores)

    return {
        "acc_overall": acc_all * 100,
        "acc_seen": acc_known * 100,
        "acc_unknown": acc_novel * 100,
        "auroc": auroc,
        "threshold": threshold
    }


if __name__ == "__main__":
    print("加载测试集 (Test Split)...")
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt"))
    test_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_super_labels.pt"))
    test_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_sub_labels.pt"))

    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt"))
    # val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
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
        device=CONFIG["device"],
        std_multiplier=CONFIG["std_multiplier"],  # 传入 k 值
        temperature=CONFIG["temperature"]
    )

    print(res)
