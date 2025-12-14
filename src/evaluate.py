import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from core.config import config, TrainingLoss, OODScoreMethod
from core.inference import predict_with_linear_single_head, predict_with_gated_dual_head, calculate_threshold_linear_single_head, calculate_threshold_gated_dual_head
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
    # ENUM-based configuration
    "training_loss": config.experiment.training_loss,
    "threshold_method": config.experiment.threshold_method,
    "prediction_method": config.experiment.prediction_method,
    "threshold_temperature": config.experiment.threshold_temperature,
    "prediction_temperature": config.experiment.prediction_temperature,
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


def run_single_trial(cfg: dict, seed: int, verbose: bool, use_val_as_test: bool):
    """
    运行单次实验，返回各项指标
    
    Args:
        cfg: 配置字典，包含以下键：
            - feature_dir, feature_dim, learning_rate, batch_size, epochs
            - novel_super_idx, novel_sub_idx, target_recall
            - enable_feature_gating, enable_hierarchical_masking
            - training_loss, threshold_method, prediction_method
            - threshold_temperature, prediction_temperature
        seed: 随机种子
        verbose: 是否打印训练进度信息
        use_val_as_test: 如果为 True，则在验证集上评估而不是测试集
    
    Returns:
        dict: 包含各项评估指标
    """
    set_seed(seed)
    
    # 加载验证和测试数据
    val_features = torch.load(os.path.join(cfg["feature_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(cfg["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(cfg["feature_dir"], "val_sub_labels.pt"))
    
    # 根据参数决定评估集
    if use_val_as_test:
        test_features = val_features.to(device)
        test_super_labels = val_super_labels
        test_sub_labels = val_sub_labels
    else:
        test_features = torch.load(os.path.join(cfg["feature_dir"], "test_features.pt")).to(device)
        test_super_labels = torch.load(os.path.join(cfg["feature_dir"], "test_super_labels.pt"))
        test_sub_labels = torch.load(os.path.join(cfg["feature_dir"], "test_sub_labels.pt"))
    
    # 使用 run_training 训练（不保存文件）
    result = run_training(
        feature_dim=cfg["feature_dim"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        device=device,
        enable_feature_gating=cfg["enable_feature_gating"],
        training_loss=cfg["training_loss"],
        feature_dir=cfg["feature_dir"],
        verbose=verbose
    )
    
    if cfg["enable_feature_gating"]:
        model, super_map, sub_map, super_to_sub = result
    else:
        super_model, sub_model, super_map, sub_map, super_to_sub = result
    
    # 创建反向映射
    super_map_inv = {v: int(k) for k, v in super_map.items()}
    sub_map_inv = {v: int(k) for k, v in sub_map.items()}
    
    # Hierarchical Masking
    if not cfg["enable_hierarchical_masking"]:
        super_to_sub = None
    
    if cfg["enable_feature_gating"]:
        # 计算阈值
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, val_features, val_super_labels, val_sub_labels,
            super_map_inv, sub_map_inv, cfg["target_recall"], device,
            cfg["threshold_temperature"], cfg["threshold_method"]
        )
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
            super_to_sub, cfg["prediction_temperature"], cfg["prediction_method"]
        )
    else:
        # 计算阈值
        thresh_super = calculate_threshold_linear_single_head(
            super_model, val_features, val_super_labels, super_map_inv,
            cfg["target_recall"], device,
            cfg["threshold_temperature"], cfg["threshold_method"]
        )
        thresh_sub = calculate_threshold_linear_single_head(
            sub_model, val_features, val_sub_labels, sub_map_inv,
            cfg["target_recall"], device,
            cfg["threshold_temperature"], cfg["threshold_method"]
        )
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
            super_to_sub, cfg["prediction_temperature"], cfg["prediction_method"]
        )
    
    # 计算指标
    super_all, super_seen, super_unseen = calculate_metrics(
        test_super_labels.numpy(), super_preds, cfg["novel_super_idx"]
    )
    sub_all, sub_seen, sub_unseen = calculate_metrics(
        test_sub_labels.numpy(), sub_preds, cfg["novel_sub_idx"]
    )
    
    # 计算 AUROC
    # 对于 AUROC，我们需要二分类标签：1=已知类, 0=未知类
    super_binary_labels = (test_super_labels.numpy() != cfg["novel_super_idx"]).astype(int)
    sub_binary_labels = (test_sub_labels.numpy() != cfg["novel_sub_idx"]).astype(int)
    
    # 计算 AUROC（得分越高，越像已知类）
    # 需要检查是否存在两个类别，否则 AUROC 无定义
    if len(np.unique(super_binary_labels)) > 1:
        super_auroc = roc_auc_score(super_binary_labels, super_scores)
    else:
        super_auroc = float('nan')
    
    if len(np.unique(sub_binary_labels)) > 1:
        sub_auroc = roc_auc_score(sub_binary_labels, sub_scores)
    else:
        sub_auroc = float('nan')
    
    return {
        "super_overall": super_all, "super_seen": super_seen, "super_unseen": super_unseen,
        "sub_overall": sub_all, "sub_seen": sub_seen, "sub_unseen": sub_unseen,
        "super_auroc": super_auroc, "sub_auroc": sub_auroc
    }


def run_multiple_trials(cfg: dict, seeds: list[int], verbose: bool, use_val_as_test: bool) -> dict:
    """
    运行多种子评估，返回聚合统计结果
    
    Args:
        cfg: 配置字典
        seeds: 随机种子列表
        verbose: 是否打印进度信息
        use_val_as_test: 如果为 True，则在验证集上评估而不是测试集
    
    Returns:
        dict: 包含均值和标准差的聚合统计结果
    """
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f">>> Trial {i+1}/{len(seeds)}, Seed={seed}")
        result = run_single_trial(cfg, seed, verbose, use_val_as_test)
        all_results.append(result)
        if verbose:
            print(f"    Subclass Unseen: {result['sub_unseen']*100:.2f}%")
    
    # 聚合统计
    stats = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        stats[f"{key}_mean"] = np.mean(values)
        stats[f"{key}_std"] = np.std(values)
    
    # 保留原始结果
    stats["raw_results"] = all_results
    
    return stats


def print_evaluation_report(stats: dict):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"  [Superclass] Overall     : {stats['super_overall_mean']*100:.2f}% ± {stats['super_overall_std']*100:.2f}%")
    print(f"  [Superclass] Seen        : {stats['super_seen_mean']*100:.2f}% ± {stats['super_seen_std']*100:.2f}%")
    print(f"  [Superclass] Unseen      : {stats['super_unseen_mean']*100:.2f}% ± {stats['super_unseen_std']*100:.2f}%")
    print(f"  [Subclass] Overall       : {stats['sub_overall_mean']*100:.2f}% ± {stats['sub_overall_std']*100:.2f}%")
    print(f"  [Subclass] Seen          : {stats['sub_seen_mean']*100:.2f}% ± {stats['sub_seen_std']*100:.2f}%")
    print(f"  [Subclass] Unseen        : {stats['sub_unseen_mean']*100:.2f}% ± {stats['sub_unseen_std']*100:.2f}%")
    print(f"  [Superclass] AUROC       : {stats['super_auroc_mean']:.4f} ± {stats['super_auroc_std']:.4f}")
    print(f"  [Subclass] AUROC         : {stats['sub_auroc_mean']:.4f} ± {stats['sub_auroc_std']:.4f}")

if __name__ == "__main__":
    USE_VAL_AS_TEST = False  # 设置为 True 在验证集上评估，False 在测试集上评估
    
    mode = "SE Feature Gating" if CONFIG["enable_feature_gating"] else "Independent Training"
    masking = "Enabled" if CONFIG["enable_hierarchical_masking"] else "Disabled"
    eval_set = "Validation Set" if USE_VAL_AS_TEST else "Test Set"
    print("=" * 75)
    print(f"Multi-seed Evaluation | Mode: {mode} | Masking: {masking} | Trials: {len(SEEDS)}")
    print("=" * 75)
    print(f"Evaluation Set: {eval_set}")
    print(f"Training Loss: {CONFIG['training_loss'].value}")
    print(f"Threshold Method: {CONFIG['threshold_method'].value} (T={CONFIG['threshold_temperature']})")
    print(f"Prediction Method: {CONFIG['prediction_method'].value} (T={CONFIG['prediction_temperature']})")
    print("=" * 75)
    
    stats = run_multiple_trials(CONFIG, SEEDS, False, USE_VAL_AS_TEST)
    print_evaluation_report(stats)
