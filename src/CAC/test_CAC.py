from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import os

from src.core.config import *
from src.core.train import set_seed, create_label_mapping
from src.CAC.CAC import *


CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "batch_size": config.experiment.batch_size,
    "target_recall": config.experiment.target_recall,
    "alpha": 10.0,
    "lambda_w": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def test_cac_openset(model, test_features, test_labels, label_map, device, target_recall):
    """
    Returns:
        dict: 包含 acc_seen (已知类精度), acc_unknown (未知类精度), auroc 的字典
    """
    model.eval()
    model.to(device)

    mapped_test_labels = torch.tensor([label_map[l.item()] for l in test_labels], dtype=torch.long)
    dataset = TensorDataset(test_features, mapped_test_labels)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    correct_seen = 0
    total_seen = 0

    all_scores = []     # 存储距离分数
    all_targets = []    # 真实标签 (-1 为未知类)

    print(f"\n开始测试 | Target Recall: {target_recall}")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            _, distances = model(inputs)

            # CAC Score 计算 (Distance * (1 - Softmin))
            softmin = F.softmax(-distances, dim=1)
            rejection_scores_vector = distances * (1 - softmin)
            min_scores, predicted = torch.min(rejection_scores_vector, dim=1)

            # 收集结果
            all_scores.extend(min_scores.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            # 计算基础准确率 (仅针对已知类)
            mask = targets != -1
            if mask.sum() > 0:
                total_seen += mask.sum().item()
                correct_seen += (predicted[mask] == targets[mask]).sum().item()

    scores_np = np.array(all_scores)
    targets_np = np.array(all_targets)

    # 拆分 已知类 和 未知类
    known_mask = (targets_np != -1)
    unknown_mask = (targets_np == -1)

    known_scores = scores_np[known_mask]
    unknown_scores = scores_np[unknown_mask]

    # --- 指标 1: 已知类准确率 (Seen Accuracy) ---
    acc_seen = 100 * correct_seen / total_seen if total_seen > 0 else 0.0

    # --- 指标 2: AUROC (整体区分度) ---
    acc_unknown = 0.0
    auroc = 0.5

    if len(unknown_scores) > 0:
        binary_labels = np.zeros_like(targets_np)
        binary_labels[known_mask] = 1

        # 分数处理:
        auroc = roc_auc_score(binary_labels, -scores_np)

        # --- 指标 3: 未知类准确率 (Unseen Accuracy / Rejection Rate) ---
        # 使target_recall% 的已知样本距离 小于 这个阈值。
        threshold = np.percentile(known_scores, target_recall * 100)

        # 判定为未知：分数 > 阈值
        correct_unknown = (unknown_scores > threshold).sum()
        acc_unknown = 100 * correct_unknown / len(unknown_scores)

    # --- 指标 4: 总体准确率 ---
    acc_overall = 100 * (correct_seen + correct_unknown) / len(targets_np)

    # 4. 返回字典
    return {
        "acc_overall": acc_overall,
        "acc_seen": acc_seen,
        "acc_unknown": acc_unknown.item(),
        "auroc": auroc
    }


if __name__ == "__main__":
    print("加载测试集 (Test Split)...")
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt"))
    test_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_super_labels.pt"))
    test_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_sub_labels.pt"))

    num_super, super_map = create_label_mapping(test_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(test_sub_labels, "sub", CONFIG["output_dir"])
    sub_map[87] = -1

    model = CACProjector(CONFIG["feature_dim"], num_sub-1, alpha=CONFIG["alpha"])
    model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], f"best_cac_model_seed_{42}.pth")))

    # ================= 测试 =================
    res = test_cac_openset(
        model=model,
        test_features=test_features,
        test_labels=test_sub_labels,
        label_map=sub_map,
        target_recall=CONFIG["target_recall"],
        device=CONFIG["device"]
    )

    print(res)
