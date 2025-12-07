import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch.nn as nn

# ================= 配置 =================
CONFIG = {
    "model_dir": "baseline_models",
    "val_data_dir": "split_data_osr",
    "feature_dim": 512,
    "novel_super_idx": 3,
    "novel_sub_idx": 87,
    "target_recall": 0.95  # 用于自动计算阈值
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 模型定义 =================
class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


# ================= 辅助函数 =================
def load_mapping_and_model(prefix):
    json_path = os.path.join(CONFIG["model_dir"], f"{prefix}_mapping.json")
    with open(json_path, 'r') as f:
        local_to_global = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(local_to_global)
    model = LinearClassifier(CONFIG["feature_dim"], num_classes)
    model.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], f"{prefix}_model.pth")))
    model.to(device)
    model.eval()
    return model, local_to_global


def calculate_threshold(model, val_features, val_labels, label_map, target_recall=0.95):
    """自动计算阈值 (仅基于已知类样本)"""
    model.eval()
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    with torch.no_grad():
        logits = model(X_known)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

    threshold = torch.quantile(max_probs, 1 - target_recall).item()
    return threshold


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
    super_model, super_map = load_mapping_and_model("superclass")
    sub_model, sub_map = load_mapping_and_model("subclass")

    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt")).to(device)
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    print("--- 2. 计算阈值 ---")
    t_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"])
    t_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"])
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