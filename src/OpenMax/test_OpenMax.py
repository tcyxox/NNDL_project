import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


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


def test_OpenMax_openset(system, test_features, test_labels, label_map):
    # 处理验证集标签
    test_y_mapped = torch.tensor([label_map[l.item()] for l in test_labels], dtype=torch.long)

    # 获取预测结果
    # system.predict 返回的类别中: 0 是未知, 1 是原类别0, 2 是原类别1 ...
    pred_indices, pred_probs = system.predict(test_features)

    # 对齐标签
    # 验证集里的未知类(-1) 应该对应预测结果的 0; 验证集里的已知类(k) 应该对应预测结果的 k+1
    gt_aligned = test_y_mapped.numpy()
    gt_aligned = np.where(gt_aligned == -1, 0, gt_aligned + 1)

    acc_all, acc_known, acc_novel = calculate_metrics(gt_aligned, pred_indices, novel_label=0) # !!!

    # # --- A. 整体准确率 (Overall Accuracy) ---
    # acc_overall = accuracy_score(gt_aligned, pred_indices) * 100
    #
    # # --- B. 已知类准确率 (Seen Accuracy) ---
    # seen_mask = gt_aligned > 0
    # if np.sum(seen_mask) > 0:
    #     acc_seen = accuracy_score(gt_aligned[seen_mask], pred_indices[seen_mask]) * 100
    # else:
    #     acc_seen = 0.0
    #
    # # --- C. 未知类准确率 (Unseen Accuracy) ---
    # unknown_mask = gt_aligned == 0
    # if np.sum(unknown_mask) > 0:
    #     acc_unknown = accuracy_score(gt_aligned[unknown_mask], pred_indices[unknown_mask]) * 100
    # else:
    #     acc_unknown = 0.0

    # --- D. AUROC (Known vs Unknown) ---
    # 构造二分类标签 (1=Known, 0=Unknown)
    binary_labels = (gt_aligned > 0).astype(int)

    # 构造二分类分数
    # pred_probs[:, 0] = P(Unknown)， P(Known) = 1 - P(Unknown)
    known_scores = 1 - pred_probs[:, 0]

    # 计算 AUROC
    if len(np.unique(binary_labels)) > 1:
        auroc = roc_auc_score(binary_labels, known_scores)
    else:
        auroc = 5.0

    # -------------------------------------------------------------
    # 4. 返回结果字典
    # -------------------------------------------------------------
    return {
        "acc_overall": acc_all * 100,
        "acc_seen": acc_known * 100,
        "acc_unknown": acc_novel * 100,
        "auroc": auroc
    }