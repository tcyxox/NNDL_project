import torch
import pandas as pd
import os

from config import *
from utils import load_mapping_and_model, calculate_threshold, predict_with_osr

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


if __name__ == "__main__":
    # --- Step 1: 加载模型和映射 ---
    print("--- Step 1: 加载模型和映射 ---")
    super_model, super_map = load_mapping_and_model("superclass", CONFIG["model_dir"], device)
    sub_model, sub_map = load_mapping_and_model("subclass", CONFIG["model_dir"], device)

    # --- Step 2: 计算阈值 ---
    print("\n--- Step 2: 计算阈值 ---")
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt"))
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    thresh_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device)
    thresh_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device)
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")

    # --- Step 3: 测试集推理 ---
    print("\n--- Step 3: 测试集推理 ---")
    test_features = torch.load(CONFIG["test_feature_path"]).to(device)
    test_image_names = torch.load(CONFIG["test_image_names"])

    super_preds, sub_preds = predict_with_osr(
        test_features, super_model, sub_model,
        super_map, sub_map,
        thresh_super, thresh_sub,
        CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device
    )

    # --- Step 4: 保存提交文件 ---
    print("\n--- Step 4: 保存提交文件 ---")
    predictions = [
        {"image": test_image_names[i], "superclass_index": super_preds[i], "subclass_index": sub_preds[i]}
        for i in range(len(test_image_names))
    ]
    df = pd.DataFrame(predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"  > 提交文件已保存至: {CONFIG['output_csv']}")