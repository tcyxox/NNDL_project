"""
统一的提交脚本 - 合并训练、调参和推理流程

运行: python submit.py

流程:
  Step 1: 划分数据 (训练集 + 验证集)
  Step 2: 用训练集训练模型
  Step 3: 在验证集上计算阈值
  Step 4: 在测试集上推理
  Step 5: 生成提交 CSV
"""
import os

import pandas as pd
import torch

from core.config import config
from core.data_split import split_features
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
    
    # 数据划分
    "novel_ratio": config.split.novel_ratio,
    "train_ratio": config.split.train_ratio,
    "val_test_ratio": config.split.val_test_ratio,
    
    # 模型参数
    "feature_dim": config.model.feature_dim,
    "batch_size": config.experiment.batch_size,
    "learning_rate": config.experiment.learning_rate,
    "epochs": config.experiment.epochs,
    # 阈值设定参数
    "threshold_method": config.experiment.threshold_method,
    "target_recall": config.experiment.target_recall,
    "std_multiplier": config.experiment.std_multiplier,
    
    # 模型选择
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "training_loss": config.experiment.training_loss,
    
    # 推理参数
    "validation_score_method": config.experiment.validation_score_method,
    "validation_score_temperature": config.experiment.validation_score_temperature,
    "prediction_score_method": config.experiment.prediction_score_method,
    "prediction_score_temperature": config.experiment.prediction_score_temperature,
    
    # OSR 标签
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    set_seed(config.experiment.seed)
    
    print("=" * 70)
    print("Unified Submission Pipeline")
    print("=" * 70)
    mode = "SE Feature Gating" if CONFIG["enable_feature_gating"] else "Independent Training"
    masking = "Enabled" if CONFIG["enable_hierarchical_masking"] else "Disabled"
    print(f"Mode: {mode} | Masking: {masking} | Device: {device}")
    print(f"Training Loss: {CONFIG['training_loss'].value}")
    print(f"Validation: {CONFIG['validation_score_method'].value} (T={CONFIG['validation_score_temperature']})")
    print(f"Prediction: {CONFIG['prediction_score_method'].value} (T={CONFIG['prediction_score_temperature']})")
    print("=" * 70)
    
    # === Step 1: 划分数据 ===
    print("\n--- Step 1: 划分数据 ---")
    
    data = split_features(
        feature_dir=CONFIG["feature_dir"],
        novel_ratio=CONFIG["novel_ratio"],
        train_ratio=CONFIG["train_ratio"],
        val_test_ratio=CONFIG["val_test_ratio"],
        novel_sub_index=CONFIG["novel_sub_idx"],
        output_dir=None,  # 不保存中间文件
        verbose=True
    )
    
    # 合并 train + val + test 作为完整训练数据
    # (因为提交时我们要用尽可能多的数据训练)
    full_train_features = torch.cat([data.train_features, data.val_features, data.test_features], dim=0)
    full_train_super = torch.cat([data.train_super_labels, data.val_super_labels, data.test_super_labels], dim=0)
    full_train_sub = torch.cat([data.train_sub_labels, data.val_sub_labels, data.test_sub_labels], dim=0)
    
    print(f"  > 完整训练样本数: {len(full_train_features)} (训练{len(data.train_features)} + 验证{len(data.val_features)} + 测试{len(data.test_features)})")
    
    # === Step 2: 训练模型 ===
    print("\n--- Step 2: 训练模型 ---")
    
    result = run_training(
        feature_dim=CONFIG["feature_dim"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        device=device,
        enable_feature_gating=CONFIG["enable_feature_gating"],
        training_loss=CONFIG["training_loss"],
        train_features=full_train_features,
        train_super_labels=full_train_super,
        train_sub_labels=full_train_sub,
        output_dir=None,  # 不保存中间文件
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
    
    # === Step 3: 计算阈值 (使用 val + test 作为阈值数据) ===
    print("\n--- Step 3: 计算阈值 ---")
    
    # 合并 val + test 用于阈值计算 (最大化阈值估计的数据量)
    threshold_features = torch.cat([data.val_features, data.test_features], dim=0)
    threshold_super_labels = torch.cat([data.val_super_labels, data.test_super_labels], dim=0)
    threshold_sub_labels = torch.cat([data.val_sub_labels, data.test_sub_labels], dim=0)
    
    print(f"  > 阈值数据量: {len(threshold_features)} (val {len(data.val_features)} + test {len(data.test_features)})")
    print(f"  > 阈值方法: {CONFIG['validation_score_method'].value} (T={CONFIG['validation_score_temperature']})")
    
    if CONFIG["enable_feature_gating"]:
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, threshold_features, threshold_super_labels, threshold_sub_labels,
            super_map_inv, sub_map_inv, device,
            CONFIG["threshold_method"], CONFIG["target_recall"], CONFIG["std_multiplier"],
            CONFIG["validation_score_temperature"], CONFIG["validation_score_method"]
        )
    else:
        thresh_super = calculate_threshold_linear_single_head(
            super_model, threshold_features, threshold_super_labels, super_map_inv, device,
            CONFIG["threshold_method"], CONFIG["target_recall"], CONFIG["std_multiplier"],
            CONFIG["validation_score_temperature"], CONFIG["validation_score_method"]
        )
        thresh_sub = calculate_threshold_linear_single_head(
            sub_model, threshold_features, threshold_sub_labels, sub_map_inv, device,
            CONFIG["threshold_method"], CONFIG["target_recall"], CONFIG["std_multiplier"],
            CONFIG["validation_score_temperature"], CONFIG["validation_score_method"]
        )
    
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")
    
    # === Step 4: 测试集推理 ===
    print("\n--- Step 4: 测试集推理 ---")
    
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_image_names = torch.load(os.path.join(CONFIG["feature_dir"], "test_image_names.pt"))
    print(f"  > 真实测试样本数: {len(test_features)}")
    print(f"  > 预测方法: {CONFIG['prediction_score_method'].value} (T={CONFIG['prediction_score_temperature']})")
    
    if CONFIG["enable_feature_gating"]:
        super_preds, sub_preds, _, _ = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    else:
        super_preds, sub_preds, _, _ = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_score_temperature"], CONFIG["prediction_score_method"]
        )
    
    # === Step 5: 生成提交文件 ===
    print("\n--- Step 5: 生成提交文件 ---")
    
    predictions = [
        {"image": test_image_names[i], "superclass_index": super_preds[i], "subclass_index": sub_preds[i]}
        for i in range(len(test_image_names))
    ]
    df = pd.DataFrame(predictions)
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

