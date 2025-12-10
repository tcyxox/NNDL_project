import torch
import os
import json
import torch.nn.functional as F

from core.config import config
from core.inference import load_linear_model, calculate_threshold, load_hierarchical_model

CONFIG = {
    "model_dir": config.paths.dev,
    "val_data_dir": config.paths.split_features,
    "target_recall": config.experiment.target_recall,
    "feature_dim": config.model.feature_dim,
    "hyperparams_file": os.path.join(config.paths.dev, "hyperparameters.json"),
    "enable_soft_attention": config.experiment.enable_soft_attention
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_threshold_hierarchical(model, val_features, val_labels, label_map, target_recall, device, is_super=True):
    """为 HierarchicalClassifier 计算阈值"""
    model.eval()
    
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)
    
    if len(X_known) == 0:
        return 0.5
    
    with torch.no_grad():
        super_logits, sub_logits = model(X_known)
        logits = super_logits if is_super else sub_logits
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
    
    threshold = torch.quantile(max_probs, 1 - target_recall).item()
    return threshold


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
    
    # --- Step 2: 加载验证集特征 ---
    print("\n--- Step 2: 加载验证集特征 ---")
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt")).to(device)
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    # --- Step 3: 计算阈值 ---
    print("\n--- Step 3: 计算阈值 ---")
    
    if CONFIG["enable_soft_attention"]:
        print("  > 使用 Soft Attention 模式")
        model, _, _ = load_hierarchical_model(
            CONFIG["model_dir"], CONFIG["feature_dim"], num_super, num_sub, True, device
        )
        thresh_super = calculate_threshold_hierarchical(model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"], device, is_super=True)
        thresh_sub = calculate_threshold_hierarchical(model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"], device, is_super=False)
    else:
        print("  > 使用独立模型模式")
        super_model, _ = load_linear_model("super", CONFIG["model_dir"], CONFIG["feature_dim"], device)
        sub_model, _ = load_linear_model("sub", CONFIG["model_dir"], CONFIG["feature_dim"], device)
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
