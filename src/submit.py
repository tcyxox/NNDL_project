"""
统一的提交脚本 - 合并训练、调参和推理流程

运行: python submit.py

流程:
  Step 1: 用完整数据训练模型
  Step 2: 在验证集上计算阈值
  Step 3: 在测试集上推理
  Step 4: 生成提交 CSV
"""
import os

import pandas as pd
import torch

from core.config import config
from core.train import run_training, create_label_mapping, create_super_to_sub_mapping
from core.inference import (
    calculate_threshold_gated_dual_head,
    calculate_threshold_linear_single_head,
    predict_with_gated_dual_head,
    predict_with_linear_single_head
)
from core.utils import set_seed

# ===================== 配置 =====================
CONFIG = {
    # 数据路径
    "feature_dir": config.paths.features,           # 完整训练数据
    "val_data_dir": config.paths.split_features,    # 验证集数据 (用于阈值计算)
    "output_csv": os.path.join(config.paths.outputs, "submission_osr.csv"),
    
    # 模型参数
    "feature_dim": config.model.feature_dim,
    "batch_size": config.experiment.batch_size,
    "learning_rate": config.experiment.learning_rate,
    "epochs": config.experiment.epochs,
    "target_recall": config.experiment.target_recall,
    
    # 模型选择
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "training_loss": config.experiment.training_loss,
    
    # 推理参数
    "threshold_method": config.experiment.threshold_method,
    "threshold_temperature": config.experiment.threshold_temperature,
    "prediction_method": config.experiment.prediction_method,
    "prediction_temperature": config.experiment.prediction_temperature,
    
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
    print(f"Threshold: {CONFIG['threshold_method'].value} (T={CONFIG['threshold_temperature']})")
    print(f"Prediction: {CONFIG['prediction_method'].value} (T={CONFIG['prediction_temperature']})")
    print("=" * 70)
    
    # === Step 1: 训练模型 (使用完整数据) ===
    print("\n--- Step 1: 训练模型 (完整数据) ---")
    
    train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))
    print(f"  > 训练样本数: {len(train_features)}")
    
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
    
    # === Step 2: 计算阈值 (使用验证集) ===
    print("\n--- Step 2: 计算阈值 (验证集) ---")
    
    val_features = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))
    print(f"  > 验证样本数: {len(val_features)}")
    print(f"  > 阈值方法: {CONFIG['threshold_method'].value} (T={CONFIG['threshold_temperature']})")
    
    if CONFIG["enable_feature_gating"]:
        thresh_super, thresh_sub = calculate_threshold_gated_dual_head(
            model, val_features, val_super_labels, val_sub_labels,
            super_map_inv, sub_map_inv, CONFIG["target_recall"], device,
            CONFIG["threshold_temperature"], CONFIG["threshold_method"]
        )
    else:
        thresh_super = calculate_threshold_linear_single_head(
            super_model, val_features, val_super_labels, super_map_inv,
            CONFIG["target_recall"], device,
            CONFIG["threshold_temperature"], CONFIG["threshold_method"]
        )
        thresh_sub = calculate_threshold_linear_single_head(
            sub_model, val_features, val_sub_labels, sub_map_inv,
            CONFIG["target_recall"], device,
            CONFIG["threshold_temperature"], CONFIG["threshold_method"]
        )
    
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")
    
    # === Step 3: 测试集推理 ===
    print("\n--- Step 3: 测试集推理 ---")
    
    test_features = torch.load(os.path.join(CONFIG["feature_dir"], "test_features.pt")).to(device)
    test_image_names = torch.load(os.path.join(CONFIG["feature_dir"], "test_image_names.pt"))
    print(f"  > 测试样本数: {len(test_features)}")
    print(f"  > 预测方法: {CONFIG['prediction_method'].value} (T={CONFIG['prediction_temperature']})")
    
    if CONFIG["enable_feature_gating"]:
        super_preds, sub_preds, _, _ = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_temperature"], CONFIG["prediction_method"]
        )
    else:
        super_preds, sub_preds, _, _ = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["prediction_temperature"], CONFIG["prediction_method"]
        )
    
    # === Step 4: 生成提交文件 ===
    print("\n--- Step 4: 生成提交文件 ---")
    
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
