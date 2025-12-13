import json
import os

import torch

from core.config import config
from core.inference import load_linear_single_head, calculate_threshold_linear_single_head, calculate_threshold_gated_dual_head, load_gated_dual_head

CONFIG = {
    "model_dir": config.paths.dev,
    "val_data_dir": config.paths.split_features,
    "target_recall": config.experiment.target_recall,
    "feature_dim": config.model.feature_dim,
    "hyperparams_file": os.path.join(config.paths.dev, "hyperparameters.json"),
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_energy": config.experiment.enable_energy,
    "enable_sigmoid_bce": config.experiment.enable_sigmoid_bce,
    "ood_temperature": config.experiment.ood_temperature,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # --- Step 1: 加载模型和映射 ---
    print("--- Step 1: 加载模型和映射 ---")
    
    # 加载映射表 (两种模式都需要)
    with open(os.path.join(CONFIG["model_dir"], "super_local_to_global_map.json"), 'r') as f:
        super_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(CONFIG["model_dir"], "sub_local_to_global_map.json"), 'r') as f:
        sub_map = {int(k): v for k, v in json.load(f).items()}
    
    num_super, num_sub = len(super_map), len(sub_map)
    print(f"  > 超类: {num_super} 个, 子类: {num_sub} 个")
    
    # 创建反向映射
    super_map_inv = {v: k for k, v in super_map.items()}
    sub_map_inv = {v: k for k, v in sub_map.items()}
    
    # --- Step 2: 加载验证集特征 ---
    print("\n--- Step 2: 加载验证集特征 ---")
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt"))
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    # --- Step 3: 计算阈值 ---
    print("\n--- Step 3: 计算阈值 ---")
    use_energy = CONFIG["enable_energy"]
    use_sigmoid_bce = CONFIG["enable_sigmoid_bce"]
    
    if CONFIG["enable_feature_gating"]:
        print("  > 使用 Soft Attention 模式")
        model, _, _ = load_gated_dual_head(
            CONFIG["model_dir"], CONFIG["feature_dim"], num_super, num_sub, device
        )
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, val_feat, val_super_lbl, val_sub_lbl, 
            super_map_inv, sub_map_inv, CONFIG["target_recall"], device,
            CONFIG["ood_temperature"], use_energy, use_sigmoid_bce
        )
    else:
        print("  > 使用独立模型模式")
        super_model, _ = load_linear_single_head("super", CONFIG["model_dir"], CONFIG["feature_dim"], device)
        sub_model, _ = load_linear_single_head("sub", CONFIG["model_dir"], CONFIG["feature_dim"], device)
        thresh_super = calculate_threshold_linear_single_head(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device, use_energy, temperature=CONFIG["ood_temperature"])
        thresh_sub = calculate_threshold_linear_single_head(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device, use_energy, temperature=CONFIG["ood_temperature"])
    
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
