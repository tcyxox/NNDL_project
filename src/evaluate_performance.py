import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from core.config import config
from core.inference import predict_with_linear_model, predict_with_hierarchical_model, calculate_threshold_linear, calculate_threshold_hierarchical
from core.train import run_training
from core.utils import set_seed

CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    "target_recall": config.experiment.target_recall,
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "enable_energy": config.experiment.enable_energy,
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
    
    # 加载验证和测试数据
    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_sub_labels.pt"))
    
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_super_labels.pt"))
    test_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "test_sub_labels.pt"))
    
    # 使用 run_training 训练（不保存文件）
    result = run_training(
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        enable_feature_gating=CONFIG["enable_feature_gating"],
        device=device,
        feature_dir=CONFIG["feature_dir"]
    )
    
    if CONFIG["enable_feature_gating"]:
        model, super_map, sub_map, super_to_sub = result
    else:
        super_model, sub_model, super_map, sub_map, super_to_sub = result
    
    # 创建反向映射
    super_map_inv = {v: int(k) for k, v in super_map.items()}
    sub_map_inv = {v: int(k) for k, v in sub_map.items()}
    
    # hierarchical masking
    if not CONFIG["enable_hierarchical_masking"]:
        super_to_sub = None
    
    use_energy = CONFIG["enable_energy"]
    
    if CONFIG["enable_feature_gating"]:
        # 计算阈值
        thresh_super, thresh_sub = calculate_threshold_hierarchical(
            model, val_features, val_super_labels, val_sub_labels,
            super_map_inv, sub_map_inv, CONFIG["target_recall"], device, use_energy
        )
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_hierarchical_model(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            use_energy, super_to_sub
        )
    else:
        # 计算阈值
        thresh_super = calculate_threshold_linear(super_model, val_features, val_super_labels, super_map_inv, CONFIG["target_recall"], device, use_energy)
        thresh_sub = calculate_threshold_linear(sub_model, val_features, val_sub_labels, sub_map_inv, CONFIG["target_recall"], device, use_energy)
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_linear_model(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            use_energy, super_to_sub
        )
    
    # 计算指标
    super_all, super_seen, super_unseen = calculate_metrics(
        test_super_labels.numpy(), super_preds, CONFIG["novel_super_idx"]
    )
    sub_all, sub_seen, sub_unseen = calculate_metrics(
        test_sub_labels.numpy(), sub_preds, CONFIG["novel_sub_idx"]
    )
    
    # 计算 AUROC
    # 对于 AUROC，我们需要二分类标签：1=已知类, 0=未知类
    super_binary_labels = (test_super_labels.numpy() != CONFIG["novel_super_idx"]).astype(int)
    sub_binary_labels = (test_sub_labels.numpy() != CONFIG["novel_sub_idx"]).astype(int)
    
    # 计算 AUROC（得分越高，越像已知类）
    try:
        super_auroc = roc_auc_score(super_binary_labels, super_scores)
        sub_auroc = roc_auc_score(sub_binary_labels, sub_scores)
    except ValueError as e:
        # 如果只有一个类别，无法计算 AUROC
        print(f"警告: 无法计算 AUROC - {e}")
        super_auroc = 0.0
        sub_auroc = 0.0
    
    return {
        "super_overall": super_all, "super_seen": super_seen, "super_unseen": super_unseen,
        "sub_overall": sub_all, "sub_seen": sub_seen, "sub_unseen": sub_unseen,
        "super_auroc": super_auroc, "sub_auroc": sub_auroc
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
    
    super_overall_list = [r["super_overall"] for r in all_results]
    super_seen_list = [r["super_seen"] for r in all_results]
    super_unseen_list = [r["super_unseen"] for r in all_results]
    sub_overall_list = [r["sub_overall"] for r in all_results]
    sub_seen_list = [r["sub_seen"] for r in all_results]
    sub_unseen_list = [r["sub_unseen"] for r in all_results]
    super_auroc_list = [r["super_auroc"] for r in all_results]
    sub_auroc_list = [r["sub_auroc"] for r in all_results]

    print(f"  [Superclass] Overall     : {np.mean(super_overall_list)*100:.2f}% ± {np.std(super_overall_list)*100:.2f}%")
    print(f"  [Superclass] Seen        : {np.mean(super_seen_list)*100:.2f}% ± {np.std(super_seen_list)*100:.2f}%")
    print(f"  [Superclass] Unseen      : {np.mean(super_unseen_list)*100:.2f}% ± {np.std(super_unseen_list)*100:.2f}%")
    print(f"  [Subclass] Overall       : {np.mean(sub_overall_list)*100:.2f}% ± {np.std(sub_overall_list)*100:.2f}%")
    print(f"  [Subclass] Seen          : {np.mean(sub_seen_list)*100:.2f}% ± {np.std(sub_seen_list)*100:.2f}%")
    print(f"  [Subclass] Unseen        : {np.mean(sub_unseen_list)*100:.2f}% ± {np.std(sub_unseen_list)*100:.2f}%")
    print(f"  [Superclass] AUROC       : {np.mean(super_auroc_list):.4f} ± {np.std(super_auroc_list):.4f}")
    print(f"  [Subclass] AUROC         : {np.mean(sub_auroc_list):.4f} ± {np.std(sub_auroc_list):.4f}")
