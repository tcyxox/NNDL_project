import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score

from core.config import config
from core.inference import load_mapping_and_model, predict_with_osr

CONFIG = {
    "hyperparams_file": os.path.join(config.paths.dev, "hyperparameters.json"),
    "model_dir": config.paths.dev,
    "test_data_dir": config.paths.split_features,
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    "enable_hierarchical_masking": config.osr.enable_hierarchical_masking,
    "feature_dim": config.model.feature_dim
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true, y_pred, novel_label, name="Task"):
    """计算详细的准确率指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc_all = accuracy_score(y_true, y_pred)

    mask_known = (y_true != novel_label)
    acc_known = accuracy_score(y_true[mask_known], y_pred[mask_known]) if np.sum(mask_known) > 0 else 0.0

    mask_novel = (y_true == novel_label)
    acc_novel = accuracy_score(y_true[mask_novel], y_pred[mask_novel]) if np.sum(mask_novel) > 0 else 0.0

    print(f"\n[{name}] 评估报告:")
    print(f"  > 总体准确率 (Overall): {acc_all * 100:.2f}%")
    print(f"  > 已知类准确率 (Seen):    {acc_known * 100:.2f}%  (目标: 保持高)")
    print(f"  > 未知类准确率 (Unseen):  {acc_novel * 100:.2f}%  (目标: >0, 越高越好)")

    return acc_all


if __name__ == "__main__":
    # --- Step 1: 加载模型和映射 ---
    print("--- Step 1: 加载模型和映射 ---")
    super_model, super_map = load_mapping_and_model("super", CONFIG["model_dir"], CONFIG["feature_dim"], device)
    sub_model, sub_map = load_mapping_and_model("sub", CONFIG["model_dir"], CONFIG["feature_dim"], device)
    
    # 加载超类到子类的映射表（用于 hierarchical masking）
    super_to_sub = None
    if CONFIG["enable_hierarchical_masking"]:
        with open(os.path.join(CONFIG["model_dir"], "super_to_sub_map.json"), 'r') as f:
            super_to_sub = {int(k): v for k, v in json.load(f).items()}
        print(f"  > Hierarchical masking 已启用")
    else:
        print(f"  > Hierarchical masking 已禁用")

    # --- Step 2: 加载阈值 ---
    print("\n--- Step 2: 加载阈值 ---")
    with open(CONFIG["hyperparams_file"], 'r') as f:
        hyperparams = json.load(f)
    thresh_super = hyperparams["thresh_super"]
    thresh_sub = hyperparams["thresh_sub"]
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")

    # --- Step 3: 在 Test 集上推理 ---
    print("\n--- Step 3: 在 Test 集上推理 ---")
    test_feat = torch.load(os.path.join(CONFIG["test_data_dir"], "test_features.pt")).to(device)
    test_super_lbl = torch.load(os.path.join(CONFIG["test_data_dir"], "test_super_labels.pt"))
    test_sub_lbl = torch.load(os.path.join(CONFIG["test_data_dir"], "test_sub_labels.pt"))

    super_preds, sub_preds = predict_with_osr(
        test_feat, super_model, sub_model,
        super_map, sub_map,
        thresh_super, thresh_sub,
        CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
        super_to_sub=super_to_sub
    )

    # --- Step 4: 评估结果 ---
    print("\n--- Step 4: 评估结果 ---")
    calculate_metrics(test_super_lbl.numpy(), super_preds, CONFIG["novel_super_idx"], "Superclass")
    calculate_metrics(test_sub_lbl.numpy(), sub_preds, CONFIG["novel_sub_idx"], "Subclass")