"""
模型评估脚本：支持多种子测试取平均
"""
import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score

from core.config import config
from core.train import create_label_mapping, train_linear_model, train_hierarchical_model, create_super_to_sub_mapping
from core.utils import set_seed
from core.models import HierarchicalClassifier
from core.inference import load_linear_model, predict_with_linear_model, load_hierarchical_model, predict_with_hierarchical_model, calculate_threshold
import torch.nn.functional as F

CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "target_recall": config.experiment.target_recall,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# 多种子评估配置
SEEDS = [42, 123, 456, 789, 1024]


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


def run_single_trial(seed):
    """运行单次实验，返回各项指标"""
    set_seed(seed)
    
    # 加载数据
    train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))
    
    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt")).to(device)
    val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_sub_labels.pt"))
    
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_super_labels.pt"))
    test_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_sub_labels.pt"))
    
    # 创建映射
    num_super, super_map = create_label_mapping(train_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", CONFIG["output_dir"])
    create_super_to_sub_mapping(train_super_labels, train_sub_labels, CONFIG["output_dir"])
    
    # 加载 super_to_sub (用于 hierarchical masking)
    super_to_sub = None
    if CONFIG["enable_hierarchical_masking"]:
        with open(os.path.join(CONFIG["output_dir"], "super_to_sub_map.json"), 'r') as f:
            super_to_sub = {int(k): v for k, v in json.load(f).items()}
    
    super_map_inv = {v: int(k) for k, v in super_map.items()}
    sub_map_inv = {v: int(k) for k, v in sub_map.items()}
    
    if CONFIG["enable_feature_gating"]:
        # 联合训练
        model = train_hierarchical_model(
            train_features, train_super_labels, train_sub_labels, super_map, sub_map, num_super, num_sub,
            CONFIG["feature_dim"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        
        # 计算阈值
        model.eval()
        with torch.no_grad():
            super_logits, sub_logits = model(val_features)
        
        known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])
        super_probs = F.softmax(super_logits[known_super], dim=1)
        thresh_super = torch.quantile(super_probs.max(dim=1)[0], 1 - CONFIG["target_recall"]).item()
        
        known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])
        sub_probs = F.softmax(sub_logits[known_sub], dim=1)
        thresh_sub = torch.quantile(sub_probs.max(dim=1)[0], 1 - CONFIG["target_recall"]).item()
        
        # 推理
        super_preds, sub_preds = predict_with_hierarchical_model(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub=super_to_sub
        )
    else:
        # 独立训练
        super_model = train_linear_model(
            train_features, train_super_labels, super_map, num_super,
            CONFIG["feature_dim"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        sub_model = train_linear_model(
            train_features, train_sub_labels, sub_map, num_sub,
            CONFIG["feature_dim"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["epochs"], device
        )
        
        # 计算阈值
        thresh_super = calculate_threshold(super_model, val_features, val_super_labels, super_map_inv, CONFIG["target_recall"], device)
        thresh_sub = calculate_threshold(sub_model, val_features, val_sub_labels, sub_map_inv, CONFIG["target_recall"], device)
        
        # 推理
        super_preds, sub_preds = predict_with_linear_model(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub=super_to_sub
        )
    
    # 计算指标
    super_all, super_seen, super_unseen = calculate_metrics(
        test_super_labels.numpy(), super_preds, CONFIG["novel_super_idx"]
    )
    sub_all, sub_seen, sub_unseen = calculate_metrics(
        test_sub_labels.numpy(), sub_preds, CONFIG["novel_sub_idx"]
    )
    
    return {
        "super_overall": super_all,
        "super_seen": super_seen,
        "super_unseen": super_unseen,
        "sub_overall": sub_all,
        "sub_seen": sub_seen,
        "sub_unseen": sub_unseen,
    }


if __name__ == "__main__":
    mode = "Soft Attention" if CONFIG["enable_feature_gating"] else "独立训练"
    masking = "启用" if CONFIG["enable_hierarchical_masking"] else "禁用"
    print("=" * 60)
    print(f"多种子评估 | 模式: {mode} | Masking: {masking} | 试验: {len(SEEDS)}次")
    print("=" * 60)
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n>>> Trial {i+1}/{len(SEEDS)}, Seed={seed}")
        result = run_single_trial(seed)
        all_results.append(result)
        print(f"    Subclass Unseen: {result['sub_unseen']*100:.2f}%")
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    
    metrics = ["super_overall", "super_seen", "super_unseen", "sub_overall", "sub_seen", "sub_unseen"]
    labels = {
        "super_overall": "[Superclass] Overall",
        "super_seen": "[Superclass] Seen",
        "super_unseen": "[Superclass] Unseen",
        "sub_overall": "[Subclass] Overall",
        "sub_seen": "[Subclass] Seen",
        "sub_unseen": "[Subclass] Unseen",
    }
    
    for m in metrics:
        vals = [r[m] for r in all_results]
        mean = np.mean(vals) * 100
        std = np.std(vals) * 100
        print(f"  {labels[m]:25s}: {mean:5.2f}% ± {std:.2f}%")
