import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm

from config import *
from utils import load_mapping_and_model, calculate_threshold

CONFIG = {
    "model_dir": MODELS_DIR,
    "val_data_dir": SPLIT_DIR,
    "test_feature_path": os.path.join(FEATURES_DIR, "test_features.pt"),
    "test_image_names": os.path.join(FEATURES_DIR, "test_image_names.pt"),
    "output_csv": os.path.join(OUTPUTS_DIR, "submission_osr.csv"),
    "novel_super_idx": NOVEL_SUPER_INDEX,
    "novel_sub_idx": NOVEL_SUB_INDEX,
    "target_recall": TARGET_RECALL
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 主程序 =================
if __name__ == "__main__":
    # --- 1. 加载模型和映射 ---
    print("--- 步骤 1: 加载模型和映射 ---")
    super_model, super_map = load_mapping_and_model("superclass", CONFIG["model_dir"], device)
    sub_model, sub_map = load_mapping_and_model("subclass", CONFIG["model_dir"], device)

    # --- 2. 利用验证集计算最佳阈值 ---
    print("\n--- 步骤 2: 计算最佳 OSR 阈值 ---")
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt"))
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    thresh_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device)
    thresh_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device)

    print(f"  > 自动计算出的 Superclass 阈值: {thresh_super:.4f}")
    print(f"  > 自动计算出的 Subclass 阈值:   {thresh_sub:.4f}")

    # --- 3. 对官方测试集进行推理 ---
    print("\n--- 步骤 3: 生成最终预测 ---")
    test_features = torch.load(CONFIG["test_feature_path"]).to(device)
    test_image_names = torch.load(CONFIG["test_image_names"])

    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(test_features)), desc="Inference"):
            feature = test_features[i].unsqueeze(0)
            image_name = test_image_names[i]

            # === 超类预测 ===
            super_logits = super_model(feature)
            super_probs = F.softmax(super_logits, dim=1)
            max_s_prob, s_idx = torch.max(super_probs, dim=1)

            if max_s_prob.item() < thresh_super:
                final_super = CONFIG["novel_super_idx"]
            else:
                final_super = super_map[s_idx.item()]

            # === 子类预测 ===
            sub_logits = sub_model(feature)
            sub_probs = F.softmax(sub_logits, dim=1)
            max_sub_prob, sub_idx = torch.max(sub_probs, dim=1)

            if max_sub_prob.item() < thresh_sub:
                final_sub = CONFIG["novel_sub_idx"]
            else:
                final_sub = sub_map[sub_idx.item()]

            # === 逻辑一致性修正 ===
            if final_super == CONFIG["novel_super_idx"]:
                final_sub = CONFIG["novel_sub_idx"]

            predictions.append({
                "image": image_name,
                "superclass_index": final_super,
                "subclass_index": final_sub
            })

    # --- 4. 保存 ---
    df = pd.DataFrame(predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"提交文件已保存至: {CONFIG['output_csv']}")