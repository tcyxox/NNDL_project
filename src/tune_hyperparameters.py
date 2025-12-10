import torch
import os
import json

from config import *
from utils import load_mapping_and_model, calculate_threshold

CONFIG = {
    "model_dir": MODELS_DIR,
    "val_data_dir": SPLIT_DIR,
    "target_recall": TARGET_RECALL,
    "hyperparams_file": os.path.join(MODELS_DIR, "hyperparameters.json")
}

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # --- Step 1: 加载模型和映射 ---
    print("--- Step 1: 加载模型和映射 ---")
    super_model, super_map = load_mapping_and_model("super", CONFIG["model_dir"], device)
    sub_model, sub_map = load_mapping_and_model("sub", CONFIG["model_dir"], device)

    # --- Step 2: 加载验证集特征 ---
    print("\n--- Step 2: 加载验证集特征 ---")
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt")).to(device)
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    # --- Step 3: 计算阈值 ---
    print("\n--- Step 3: 计算阈值 ---")
    thresh_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device)
    thresh_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device)
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")

    # --- Step 4: 保存超参数 ---
    print("\n--- Step 4: 保存超参数 ---")
    hyperparams = {
        "thresh_super": thresh_super,
        "thresh_sub": thresh_sub
    }
    
    with open(CONFIG["hyperparams_file"], 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"  > 超参数已保存到: {CONFIG['hyperparams_file']}")
