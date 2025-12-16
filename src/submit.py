"""
统一的提交脚本 - 合并训练、调参和推理流程

流程:
  Phase 1: 多种子阈值校准
    - 对每个种子: 划分数据 -> 在 train 上训练 -> 在 val+test 上计算阈值
    - 取平均阈值
  Phase 2: 训练最终模型 (全量数据)
  Phase 3: 使用平均阈值在真实测试集上推理

潜在问题：在略小数据集上训练的模型能力与在全量数据集上训练的模型能力存在差异，可能导致小幅度的阈值偏移
"""
import os

import numpy as np
import pandas as pd
import torch

from core.config import config
from core.data_split import load_full_dataset, split_train_to_subtrain_val
from core.training import run_training
from core.prediction import (
    predict_with_gated_dual_head,
    predict_with_linear_single_head
)
from core.validation import (
    calculate_threshold_gated_dual_head,
    calculate_threshold_linear_single_head
)
from core.utils import set_seed

# ===================== 配置 =====================
CONFIG = {
    # 数据路径
    "feature_dir": config.paths.features,
    "output_csv": os.path.join(config.paths.outputs, "submission_osr.csv"),
    
    # 验证集划分 (Inner Loop Params)
    "val_include_novel": config.experiment.val_include_novel,
    "force_super_novel": config.experiment.force_super_novel,
    "val_ratio": config.split.val_ratio,
    "val_sub_novel_ratio": config.split.val_sub_novel_ratio,
    
    # 模型参数
    "feature_dim": config.model.feature_dim,
    "batch_size": config.experiment.batch_size,
    "learning_rate": config.experiment.learning_rate,
    "epochs": config.experiment.epochs,
    
    # 模型选择
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "training_loss": config.experiment.training_loss,
    
    # 阈值设定参数（自动选择）
    "known_only_threshold": config.experiment.known_only_threshold,
    "full_val_threshold": config.experiment.full_val_threshold,
    "target_recall": config.experiment.target_recall,
    "std_multiplier": config.experiment.std_multiplier,
    
    # 推理参数
    "validation_score_method": config.experiment.validation_score_method,
    "validation_score_temperature": config.experiment.validation_score_temperature,
    "prediction_score_method": config.experiment.prediction_score_method,
    "prediction_score_temperature": config.experiment.prediction_score_temperature,
    
    # OSR 标签
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
}

# 多种子阈值校准配置
CALIBRATION_SEEDS = [42, 123, 456, 789, 1024]

device = "cuda" if torch.cuda.is_available() else "cpu"


def calibrate_threshold_single_seed(cfg: dict, seed: int):
    """
    单次阈值校准: 
    1. Load Full Data (treated as Inner Train)
    2. Split to SubTrain + Val (Inner Split)
    3. Train on SubTrain, Calibrate on Val
    
    Returns:
        (thresh_super, thresh_sub): 超类和子类阈值
    """
    set_seed(seed)
    
    # 0. Load Full Dataset
    full_dataset = load_full_dataset(cfg["feature_dir"])

    # 1. 划分数据 (Simulate Inner Loop)
    # Full Data -> SubTrain + Val
    split_info = split_train_to_subtrain_val(
        train_dataset=full_dataset,
        val_ratio=cfg["val_ratio"],
        val_sub_novel_ratio=cfg["val_sub_novel_ratio"],
        val_include_novel=cfg["val_include_novel"],
        novel_sub_index=cfg["novel_sub_idx"],
        novel_super_index=cfg["novel_super_idx"],
        force_super_novel=cfg["force_super_novel"],
        seed=seed,
        verbose=False
    )
    
    train_set = split_info.train_set # SubTrain
    val_set = split_info.test_set    # Val

    # 2. 在 SubTrain 上训练模型
    result = run_training(
        feature_dim=cfg["feature_dim"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        device=device,
        enable_feature_gating=cfg["enable_feature_gating"],
        training_loss=cfg["training_loss"],
        train_features=train_set.features,
        train_super_labels=train_set.super_labels,
        train_sub_labels=train_set.sub_labels,
        output_dir=None,
        verbose=False
    )
    
    if cfg["enable_feature_gating"]:
        model, super_map, sub_map, super_to_sub = result
    else:
        super_model, sub_model, super_map, sub_map, super_to_sub = result
    
    # 创建反向映射
    super_map_inv = {v: int(k) for k, v in super_map.items()}
    sub_map_inv = {v: int(k) for k, v in sub_map.items()}
    
    # 3. 在 Val 上计算阈值
    # val_set contains novel classes (if val_include_novel=True)
    
    # Check if we should use FullVal method or KnownOnly method
    if cfg["enable_feature_gating"]:
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, val_set.features, val_set.super_labels, val_set.sub_labels,
            super_map_inv, sub_map_inv, device,
            cfg["val_include_novel"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
    else:
        thresh_super = calculate_threshold_linear_single_head(
            super_model, val_set.features, val_set.super_labels, super_map_inv, device,
            cfg["val_include_novel"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
        thresh_sub = calculate_threshold_linear_single_head(
            sub_model, val_set.features, val_set.sub_labels, sub_map_inv, device,
            cfg["val_include_novel"],
            cfg["known_only_threshold"], cfg["full_val_threshold"],
            cfg["target_recall"], cfg["std_multiplier"],
            cfg["validation_score_temperature"], cfg["validation_score_method"]
        )
    
    return thresh_super, thresh_sub


if __name__ == "__main__":
    print("=" * 70)
    print("Unified Submission Pipeline (Multi-seed Threshold Calibration)")
    print("=" * 70)
    mode = "SE Feature Gating" if CONFIG["enable_feature_gating"] else "Independent Training"
    masking = "Enabled" if CONFIG["enable_hierarchical_masking"] else "Disabled"
    print(f"Mode: {mode} | Masking: {masking} | Device: {device}")
    print(f"Training Loss: {CONFIG['training_loss'].value}")
    print(f"Validation: {CONFIG['validation_score_method'].value} (T={CONFIG['validation_score_temperature']})")
    print(f"Prediction: {CONFIG['prediction_score_method'].value} (T={CONFIG['prediction_score_temperature']})")
    print("=" * 70)
    
    # === Phase 1: 多种子阈值校准 ===
    print("\n" + "=" * 50)
    print("Phase 1: Multi-seed Threshold Calibration")
    print("=" * 50)
    
    all_thresh_super = []
    all_thresh_sub = []
    
    for i, seed in enumerate(CALIBRATION_SEEDS):
        print(f"\n>>> Calibration Trial {i+1}/{len(CALIBRATION_SEEDS)}, Seed={seed}")
        thresh_super, thresh_sub = calibrate_threshold_single_seed(CONFIG, seed)
        all_thresh_super.append(thresh_super)
        all_thresh_sub.append(thresh_sub)
        print(f"    Super: {thresh_super:.4f}, Sub: {thresh_sub:.4f}")
    
    # 取平均阈值
    avg_thresh_super = np.mean(all_thresh_super)
    avg_thresh_sub = np.mean(all_thresh_sub)
    std_thresh_super = np.std(all_thresh_super)
    std_thresh_sub = np.std(all_thresh_sub)
    
    print(f"\n>>> 阈值统计:")
    print(f"    Superclass: {avg_thresh_super:.4f} ± {std_thresh_super:.4f}")
    print(f"    Subclass:   {avg_thresh_sub:.4f} ± {std_thresh_sub:.4f}")
    
    # === Phase 2: 训练最终模型 (全量数据) ===
    print("\n" + "=" * 50)
    print("Phase 2: Train Final Model (Full Data)")
    print("=" * 50)
    
    set_seed(config.experiment.seed)
    
    # 加载完整训练数据 (不做划分)
    # 加载完整训练数据 (不做划分)
    full_dataset = load_full_dataset(CONFIG["feature_dir"])
    train_features = full_dataset.features
    train_super_labels = full_dataset.super_labels
    train_sub_labels = full_dataset.sub_labels
    
    print(f"  > 完整训练样本数: {len(train_features)}")
    
    result = run_training(
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        device=device,
        enable_feature_gating=CONFIG["enable_feature_gating"],
        training_loss=CONFIG["training_loss"],
        train_features=train_features,
        train_super_labels=train_super_labels,
        train_sub_labels=train_sub_labels,
        output_dir=None,
        verbose=True
    )
    
    if CONFIG["enable_feature_gating"]:
        model, super_map, sub_map, super_to_sub = result
    else:
        super_model, sub_model, super_map, sub_map, super_to_sub = result
    
    # 创建反向映射
    super_map_inv = {v: int(k) for k, v in super_map.items()}
    sub_map_inv = {v: int(k) for k, v in sub_map.items()}
    
    # Hierarchical Masking
    if not CONFIG["enable_hierarchical_masking"]:
        super_to_sub = None

    # === Phase 2.5: 在全量训练集上评估模型 (Sanity Check) ===
    print("\n" + "=" * 50)
    print("Phase 2.5: Evaluate Final Model on Training Set (Sanity Check)")
    print("=" * 50)
    
    # 预测训练集
    if CONFIG["enable_feature_gating"]:
        train_super_preds, train_sub_preds, _, _ = predict_with_gated_dual_head(
            train_features.to(device), model, super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    else:
        train_super_preds, train_sub_preds, _, _ = predict_with_linear_single_head(
            train_features.to(device), super_model, sub_model,
            super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    
    # 计算准确率 (Train set should be 100% known)
    from sklearn.metrics import accuracy_score
    train_super_acc = accuracy_score(train_super_labels.numpy(), train_super_preds)
    train_sub_acc = accuracy_score(train_sub_labels.numpy(), train_sub_preds)
    
    # 统计被误判为 Novel 的比例
    train_super_novel_ratio = np.mean(np.array(train_super_preds) == CONFIG["novel_super_idx"])
    train_sub_novel_ratio = np.mean(np.array(train_sub_preds) == CONFIG["novel_sub_idx"])
    
    print(f"  > Training Superclass Accuracy: {train_super_acc*100:.2f}%")
    print(f"  > Training Subclass Accuracy:   {train_sub_acc*100:.2f}%")
    print(f"  > Training Superclass Novel Ratio (Mistake): {train_super_novel_ratio*100:.2f}%")
    print(f"  > Training Subclass Novel Ratio (Mistake):   {train_sub_novel_ratio*100:.2f}%")
    
    # === Phase 3: 使用平均阈值在真实测试集上推理 ===
    print("\n" + "=" * 50)
    print("Phase 3: Inference with Calibrated Threshold")
    print("=" * 50)
    
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_image_names = torch.load(os.path.join(CONFIG["feature_dir"], "test_image_names.pt"))
    print(f"  > 真实测试样本数: {len(test_features)}")
    print(f"  > 使用阈值: Super={avg_thresh_super:.4f}, Sub={avg_thresh_sub:.4f}")
    print(f"  > 预测方法: {CONFIG['prediction_score_method'].value} (T={CONFIG['prediction_score_temperature']})")
    
    if CONFIG["enable_feature_gating"]:
        super_preds, sub_preds, _, _ = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    else:
        super_preds, sub_preds, _, _ = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    
    # === 生成提交文件 ===
    print("\n--- Generating Submission File ---")
    
    predictions = []
    for i in range(len(test_image_names)):
        img_name = test_image_names[i]
        predictions.append({
            "image": img_name,
            "superclass_index": super_preds[i],
            "subclass_index": sub_preds[i],
            # 用于排序的辅助列
            # 用于排序的辅助列
            "_sort_idx": int(img_name.split('.')[0]) if img_name.split('.')[0].isdigit() else float('inf')
        })
    
    # 根据辅助列排序
    predictions.sort(key=lambda x: x["_sort_idx"])
    
    # 移除辅助列并创建 DataFrame
    final_predictions = []
    for p in predictions:
        final_predictions.append({
            "image": p["image"],
            "superclass_index": p["superclass_index"],
            "subclass_index": p["subclass_index"]
        })
        
    df = pd.DataFrame(final_predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    
    print(f"  > 提交文件已保存至: {CONFIG['output_csv']}")
    print(f"  > 共 {len(predictions)} 条预测")
    
    # 统计信息
    novel_super_count = sum(1 for p in super_preds if p == CONFIG["novel_super_idx"])
    novel_sub_count = sum(1 for p in sub_preds if p == CONFIG["novel_sub_idx"])
    print(f"  > Novel superclass 预测数: {novel_super_count}")
    print(f"  > Novel subclass 预测数: {novel_sub_count}")
    
    print("\n" + "=" * 70)
    print("提交完成!")
    print("=" * 70)
