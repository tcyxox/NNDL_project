import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config import *
from utils import LinearClassifier, load_mapping_and_model, calculate_threshold

CONFIG = {
    "model_dir": MODELS_DIR,
    "val_data_dir": SPLIT_DIR,
    "novel_super_idx": NOVEL_SUPER_INDEX,
    "novel_sub_idx": NOVEL_SUB_INDEX,
    "target_recall": TARGET_RECALL
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true, y_pred, novel_label, name="Task"):
    """计算详细的准确率指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. 总体准确率 (Overall Accuracy)
    acc_all = accuracy_score(y_true, y_pred)

    # 2. 已知类准确率 (Seen Accuracy)
    mask_known = (y_true != novel_label)
    if np.sum(mask_known) > 0:
        acc_known = accuracy_score(y_true[mask_known], y_pred[mask_known])
    else:
        acc_known = 0.0

    # 3. 未知类准确率 (Unseen/Novel Accuracy)
    mask_novel = (y_true == novel_label)
    if np.sum(mask_novel) > 0:
        acc_novel = accuracy_score(y_true[mask_novel], y_pred[mask_novel])
    else:
        acc_novel = 0.0

    print(f"\n[{name}] 评估报告:")
    print(f"  > 总体准确率 (Overall): {acc_all * 100:.2f}%")
    print(f"  > 已知类准确率 (Seen):    {acc_known * 100:.2f}%  (目标: 保持高)")
    print(f"  > 未知类准确率 (Unseen):  {acc_novel * 100:.2f}%  (目标: >0, 越高越好)")

    return acc_all


# ================= 主程序 =================
if __name__ == "__main__":
    print("--- 1. 加载资源 ---")
    super_model, super_map = load_mapping_and_model("superclass", CONFIG["model_dir"], device)
    sub_model, sub_map = load_mapping_and_model("subclass", CONFIG["model_dir"], device)

    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt")).to(device)
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    print("--- 2. 计算阈值 ---")
    t_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device)
    t_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device)
    print(f"  Super Threshold: {t_super:.4f}")
    print(f"  Sub Threshold:   {t_sub:.4f}")

    print("--- 3. 验证集推理 ---")
    super_preds = []
    sub_preds = []

    with torch.no_grad():
        for i in tqdm(range(len(val_feat))):
            feat = val_feat[i].unsqueeze(0)

            # --- Superclass ---
            s_logits = super_model(feat)
            s_probs = F.softmax(s_logits, dim=1)
            max_s, s_idx = torch.max(s_probs, dim=1)

            if max_s.item() < t_super:
                final_s = CONFIG["novel_super_idx"]
            else:
                final_s = super_map[s_idx.item()]

            # --- Subclass ---
            sub_logits = sub_model(feat)
            sub_probs = F.softmax(sub_logits, dim=1)
            max_sub, sub_idx = torch.max(sub_probs, dim=1)

            if max_sub.item() < t_sub:
                final_sub = CONFIG["novel_sub_idx"]
            else:
                final_sub = sub_map[sub_idx.item()]

            # --- Logic Consistency ---
            if final_s == CONFIG["novel_super_idx"]:
                final_sub = CONFIG["novel_sub_idx"]

            super_preds.append(final_s)
            sub_preds.append(final_sub)

    print("--- 4. 最终结果 ---")

    # 评估超类
    calculate_metrics(val_super_lbl.numpy(), super_preds, CONFIG["novel_super_idx"], "Superclass")

    # 评估子类
    calculate_metrics(val_sub_lbl.numpy(), sub_preds, CONFIG["novel_sub_idx"], "Subclass")