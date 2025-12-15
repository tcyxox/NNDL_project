import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from core.config import config, TrainingLoss, OODScoreMethod, KnownOnlyThreshold, FullValThreshold
from core.data_split import split_features
from core.prediction import predict_with_linear_single_head, predict_with_gated_dual_head
from core.validation import calculate_threshold_linear_single_head, calculate_threshold_gated_dual_head
from core.training import run_training
from core.utils import set_seed

CONFIG = {
    "feature_dir": config.paths.features,  # 原始特征目录（用于 split_features）
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    # 模型参数
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    # 阈值设定参数（自动选择）
    "known_only_threshold": config.experiment.known_only_threshold,
    "full_val_threshold": config.experiment.full_val_threshold,
    "target_recall": config.experiment.target_recall,
    "std_multiplier": config.experiment.std_multiplier,
    # 方法参数
    "training_loss": config.experiment.training_loss,
    "validation_score_method": config.experiment.validation_score_method,
    "prediction_score_method": config.experiment.prediction_score_method,
    "validation_score_temperature": config.experiment.validation_score_temperature,
    "prediction_score_temperature": config.experiment.prediction_score_temperature,
    # 数据划分参数
    "test_only_unknown": config.experiment.test_only_unknown,
    "novel_ratio": config.split.novel_ratio,
    "train_ratio": config.split.train_ratio,
    "val_test_ratio": config.split.val_test_ratio,
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


def run_single_trial(cfg: dict, seed: int, verbose: bool):
    """
    运行单次实验，返回各项指标
    
    Args:
        cfg: 配置字典
        seed: 随机种子（控制数据划分、模型初始化和训练）
        verbose: 是否打印训练进度信息
    Returns:
        dict: 包含各项评估指标
    """
    # 1. 数据划分
    set_seed(seed)
    
    data = split_features(
        feature_dir=cfg["feature_dir"],
        novel_ratio=cfg["novel_ratio"],
        train_ratio=cfg["train_ratio"],
        val_test_ratio=cfg["val_test_ratio"],
        test_only_unknown=cfg["test_only_unknown"],
        novel_sub_index=cfg["novel_sub_idx"],
        novel_super_index=cfg["novel_super_idx"],
        verbose=verbose
    )
    
    # 2. 设置训练种子
    set_seed(seed)
    
    # 获取数据
    train_features = data.train_features
    train_super_labels = data.train_super_labels
    train_sub_labels = data.train_sub_labels
    val_features = data.val_features
    val_super_labels = data.val_super_labels
    val_sub_labels = data.val_sub_labels
    test_features = data.test_features.to(device)
    test_super_labels = data.test_super_labels
    test_sub_labels = data.test_sub_labels
    
    # 3. 训练模型（使用 train 数据）
    result = run_training(
        feature_dim=cfg["feature_dim"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        device=device,
        enable_feature_gating=cfg["enable_feature_gating"],
        training_loss=cfg["training_loss"],
        train_features=train_features,
        train_super_labels=train_super_labels,
        train_sub_labels=train_sub_labels,
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
        # 计算阈值（根据 test_only_unknown 选择方法）
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, val_features, val_super_labels, val_sub_labels,
            super_map_inv, sub_map_inv, device,
            cfg["test_only_unknown"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
            super_to_sub, cfg["prediction_score_temperature"], cfg["prediction_score_method"]
        )
    else:
        # 计算阈值（根据 test_only_unknown 选择方法）
        thresh_super = calculate_threshold_linear_single_head(
            super_model, val_features, val_super_labels, super_map_inv, device,
            cfg["test_only_unknown"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
        thresh_sub = calculate_threshold_linear_single_head(
            sub_model, val_features, val_sub_labels, sub_map_inv, device,
            cfg["test_only_unknown"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
        
        # 推理
        super_preds, sub_preds, super_scores, sub_scores = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
            super_to_sub, cfg["prediction_score_temperature"], cfg["prediction_score_method"]
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


def run_multiple_trials(cfg: dict, seeds: list[int], verbose: bool) -> dict:
    """
    运行多种子评估，返回聚合统计结果
    
    Args:
        cfg: 配置字典
        seeds: 随机种子列表
        verbose: 是否打印进度信息
    Returns:
        dict: 包含均值和标准差的聚合统计结果
    """
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(seeds)} | Seed: {seed}")
        print("="*60)
        result = run_single_trial(cfg, seed, verbose)
        all_results.append(result)
        
        # 打印单次结果摘要
        print(f"\n[Trial Result]")
        print(f"  Subclass Overall: {result['sub_overall']*100:.2f}%")
        print(f"  Subclass Seen/Unseen: {result['sub_seen']*100:.2f}% / {result['sub_unseen']*100:.2f}%")
    
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
    print("Final Evaluation Report")
    print("=" * 60)
    print(f"  [Superclass] Overall     : {stats['super_overall_mean']*100:.2f}% ± {stats['super_overall_std']*100:.2f}%")
    print(f"  [Superclass] Seen        : {stats['super_seen_mean']*100:.2f}% ± {stats['super_seen_std']*100:.2f}%")
    print(f"  [Superclass] Unseen      : {stats['super_unseen_mean']*100:.2f}% ± {stats['super_unseen_std']*100:.2f}%")
    print(f"  [Subclass] Overall       : {stats['sub_overall_mean']*100:.2f}% ± {stats['sub_overall_std']*100:.2f}%")
    print(f"  [Subclass] Seen          : {stats['sub_seen_mean']*100:.2f}% ± {stats['sub_seen_std']*100:.2f}%")
    print(f"  [Subclass] Unseen        : {stats['sub_unseen_mean']*100:.2f}% ± {stats['sub_unseen_std']*100:.2f}%")
    print(f"  [Superclass] AUROC       : {stats['super_auroc_mean']:.4f} ± {stats['super_auroc_std']:.4f}")
    print(f"  [Subclass] AUROC         : {stats['sub_auroc_mean']:.4f} ± {stats['sub_auroc_std']:.4f}")
    
    # 一行摘要，方便复制到 eval.md
    print("\n[Copy to eval.md]")
    print(f"{stats['super_seen_mean']*100:.2f}% ± {stats['super_seen_std']*100:.2f}%, "
          f"{stats['sub_overall_mean']*100:.2f}% ± {stats['sub_overall_std']*100:.2f}%, "
          f"{stats['sub_seen_mean']*100:.2f}% ± {stats['sub_seen_std']*100:.2f}%, "
          f"{stats['sub_unseen_mean']*100:.2f}% ± {stats['sub_unseen_std']*100:.2f}%, "
          f"{stats['sub_auroc_mean']:.4f} ± {stats['sub_auroc_std']:.4f}")


def print_global_config(cfg: dict, seeds: list[int]):
    """打印全局配置"""
    mode = "SE Feature Gating" if cfg["enable_feature_gating"] else "Independent Training"
    masking = "Enabled" if cfg["enable_hierarchical_masking"] else "Disabled"
    split_mode = "Test Only" if cfg["test_only_unknown"] else "Val + Test"
    
    print("=" * 60)
    print("Global Configuration")
    print("=" * 60)
    
    print("\n[Model]")
    print(f"  Mode: {mode}")
    print(f"  Hierarchical Masking: {masking}")
    
    print("\n[Training]")
    print(f"  Loss: {cfg['training_loss'].value.upper()}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch Size: {cfg['batch_size']}")
    print(f"  Learning Rate: {cfg['learning_rate']}")
    
    print("\n[Data Split]")
    print(f"  Unknown in: {split_mode}")
    print(f"  Novel Ratio (per split): {cfg['novel_ratio']*100:.0f}%")
    print(f"  Train Ratio: {cfg['train_ratio']*100:.0f}%")
    
    print("\n[Threshold]")
    if cfg["test_only_unknown"]:
        print(f"  Method: {cfg['known_only_threshold'].value} (val has no unknown)")
    else:
        print(f"  Method: {cfg['full_val_threshold'].value} (val has unknown)")
    
    print("\n[OOD Score]")
    print(f"  Validation: {cfg['validation_score_method'].value} (T={cfg['validation_score_temperature']})")
    print(f"  Prediction: {cfg['prediction_score_method'].value} (T={cfg['prediction_score_temperature']})")
    
    print("\n[Evaluation]")
    print(f"  Seeds: {seeds}")
    print(f"  Total Trials: {len(seeds)}")


if __name__ == "__main__":
    verbose = False
    
    print_global_config(CONFIG, SEEDS)
    
    stats = run_multiple_trials(CONFIG, SEEDS, verbose)
    print_evaluation_report(stats)

