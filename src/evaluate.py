"""
嵌套验证评估脚本 (Nested Validation Evaluation)

流程 (模拟真实提交场景):
  外层循环 (Outer Loop): 生成最终报告
    1. 从 Full Train 划分出 Train (Pure Known) + Test (Mix with Sub Novel)
    
    内层循环 (Inner Loop): 阈值校准
      2.1 从 Train 划分出 SubTrain (Pure Known) + Val (Mix with Super+Sub Novel)
      2.2 在 SubTrain 上训练
      2.3 在 Val 上计算阈值
    3. 取平均阈值
    
    4. 在 Train 上训练最终模型
    5. 在 Test 上推理并记录统计

  外层循环汇总多个 seed 的结果，生成最终报告
"""
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from core.config import config
from core.dataset import (
    load_full_dataset,
    split_full_train_to_train_test,
    split_train_to_subtrain_val
)
from core.prediction import predict_with_linear_single_head, predict_with_gated_dual_head, predict_with_openmax
from core.calibration import calculate_threshold_linear_single_head, calculate_threshold_gated_dual_head, calibrate_openmax_threshold_eer, _find_openmax_eer_threshold
from core.training import run_training
from core.openmax import OpenMax
from core.utils import set_seed

# ===================== 配置 =====================
CONFIG = {
    # 数据路径
    "feature_dir": config.paths.features,
    "output_dir": config.paths.dev,
    
    # 模型参数
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    
    # 模型选择
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    
    # 阈值设定参数
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
    
    # OSR 标签
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    
    # Outer Split: Full -> Train (Pure Known) + Test (Mix)
    "test_ratio": config.split.test_ratio,
    "test_sub_novel_ratio": config.split.test_sub_novel_ratio,
    
    # Inner Split: Train -> SubTrain (Pure Known) + Val (Mix)
    "val_ratio": config.split.val_ratio,
    "val_sub_novel_ratio": config.split.val_sub_novel_ratio,
    "val_include_novel": config.experiment.val_include_novel,
    "force_super_novel": config.experiment.force_super_novel,
    
    # OpenMax 配置
    "enable_openmax": config.experiment.enable_openmax,
    "openmax_config": config.openmax,
    
    # 其他
    "verbose": config.experiment.verbose,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# 多种子评估配置
OUTER_SEEDS = [42, 123, 456, 789, 1024]
INNER_SEEDS = [42, 123, 456, 789, 1024]


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


def calibrate_threshold_inner_loop(cfg: dict, train_dataset, seeds: list[int]):
    """
    Inner loop: Iterate through each superclass as novel, calculate average threshold
    
    Args:
        cfg: Config dict
        train_dataset: Train dataset from outer split (Pure Known)
        seeds: Seeds for randomization (used cyclically)
        
    Returns:
        (avg_thresh_super, avg_thresh_sub): Average thresholds
    """
    all_thresh_super = []
    all_thresh_sub = []
    
    # Get all unique superclasses in train_dataset
    all_supers = sorted(torch.unique(train_dataset.super_labels).tolist())
    n_supers = len(all_supers)
    
    for i, target_super in enumerate(all_supers):
        # Use seeds cyclically for randomization
        seed = seeds[i % len(seeds)]
        set_seed(seed)
        
        # 1. Split Train -> SubTrain + Val, with target_super as novel
        inner_split = split_train_to_subtrain_val(
            train_dataset=train_dataset,
            val_ratio=cfg["val_ratio"],
            val_sub_novel_ratio=cfg["val_sub_novel_ratio"],
            val_include_novel=cfg["val_include_novel"],
            force_super_novel=cfg["force_super_novel"],
            target_super_novel=target_super,  # Specify which super to use as novel
            novel_sub_index=cfg["novel_sub_idx"],
            novel_super_index=cfg["novel_super_idx"],
            seed=seed,
            verbose=cfg["verbose"]
        )
        
        subtrain_set = inner_split.train_set
        val_set = inner_split.test_set
        
        # 2. Train on SubTrain
        result = run_training(
            feature_dim=cfg["feature_dim"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            device=device,
            enable_feature_gating=cfg["enable_feature_gating"],
            training_loss=cfg["training_loss"],
            train_features=subtrain_set.features,
            train_super_labels=subtrain_set.super_labels,
            train_sub_labels=subtrain_set.sub_labels,
            output_dir=None,
            verbose=cfg["verbose"]
        )
        
        if cfg["enable_feature_gating"]:
            model, super_map, sub_map, super_to_sub = result
        else:
            super_model, sub_model, super_map, sub_map, super_to_sub = result
        
        # Create inverse mapping
        super_map_inv = {v: int(k) for k, v in super_map.items()}
        sub_map_inv = {v: int(k) for k, v in sub_map.items()}
        
        # 3. Calculate threshold on Val
        if cfg["enable_openmax"]:
            # OpenMax 分支：拟合 Weibull + EER 阈值
            if cfg["enable_feature_gating"]:
                # 拟合 OpenMax for super
                openmax_super = OpenMax(
                    num_classes=len(super_map),
                    weibull_tail_size=cfg["openmax_config"].weibull_tail_size,
                    alpha=cfg["openmax_config"].alpha,
                    distance_type=cfg["openmax_config"].distance_type
                )
                with torch.no_grad():
                    super_logits, _ = model(subtrain_set.features.to(device))
                super_labels_local = torch.tensor([super_map[l.item()] for l in subtrain_set.super_labels])
                openmax_super.fit(super_logits, super_labels_local)
                
                # 拟合 OpenMax for sub
                openmax_sub = OpenMax(
                    num_classes=len(sub_map),
                    weibull_tail_size=cfg["openmax_config"].weibull_tail_size,
                    alpha=cfg["openmax_config"].alpha,
                    distance_type=cfg["openmax_config"].distance_type
                )
                with torch.no_grad():
                    _, sub_logits = model(subtrain_set.features.to(device))
                sub_labels_local = torch.tensor([sub_map[l.item()] for l in subtrain_set.sub_labels])
                openmax_sub.fit(sub_logits, sub_labels_local)
                
                # EER 阈值校准
                known_super_set = set(super_map_inv.values())
                known_sub_set = set(sub_map_inv.values())
                
                # 获取 super logits 并计算阈值
                with torch.no_grad():
                    val_super_logits, val_sub_logits = model(val_set.features.to(device))
                
                super_probs = openmax_super.predict(val_super_logits)
                super_unknown_probs = super_probs[:, 0]
                known_super_mask = np.array([l in known_super_set for l in val_set.super_labels.numpy()])
                if known_super_mask.sum() > 0 and (~known_super_mask).sum() > 0:
                    thresh_super = _find_openmax_eer_threshold(
                        super_unknown_probs[known_super_mask],
                        super_unknown_probs[~known_super_mask]
                    )
                else:
                    thresh_super = 0.5
                
                sub_probs = openmax_sub.predict(val_sub_logits)
                sub_unknown_probs = sub_probs[:, 0]
                known_sub_mask = np.array([l in known_sub_set for l in val_set.sub_labels.numpy()])
                if known_sub_mask.sum() > 0 and (~known_sub_mask).sum() > 0:
                    thresh_sub = _find_openmax_eer_threshold(
                        sub_unknown_probs[known_sub_mask],
                        sub_unknown_probs[~known_sub_mask]
                    )
                else:
                    thresh_sub = 0.5
            else:
                raise NotImplementedError("OpenMax 目前仅支持 enable_feature_gating=True 模式")
        elif cfg["enable_feature_gating"]:
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
        
        all_thresh_super.append(thresh_super)
        all_thresh_sub.append(thresh_sub)
    
    # Return average threshold
    return np.mean(all_thresh_super), np.mean(all_thresh_sub)



def run_single_outer_trial(cfg: dict, outer_seed: int, inner_seeds: list[int]):
    """
    运行单次外层实验 (Outer Trial)
    
    流程:
      1. 外层划分: Full -> Train + Test
      2. 内层循环: 阈值校准
      3. 在 Train 上训练最终模型
      4. 在 Test 上推理
    
    Args:
        cfg: 配置字典
        outer_seed: 外层种子
        inner_seeds: 内层种子列表
        
    Returns:
        dict: 包含各项评估指标
    """
    # 1. 外层划分: Full -> Train (Pure Known) + Test (Mix)
    set_seed(outer_seed)
    
    full_dataset = load_full_dataset(cfg["feature_dir"])
    
    outer_split = split_full_train_to_train_test(
        full_dataset=full_dataset,
        test_ratio=cfg["test_ratio"],
        test_sub_novel_ratio=cfg["test_sub_novel_ratio"],
        novel_sub_index=cfg["novel_sub_idx"],
        seed=outer_seed,
        verbose=cfg["verbose"]
    )
    
    train_dataset = outer_split.train_set  # Pure Known
    test_dataset = outer_split.test_set    # Mix (Known + Novel)
    
    # 2. 内层循环: 阈值校准
    if cfg["verbose"]:
        print(f"\n  [Inner Loop] Calibrating threshold with {len(inner_seeds)} seeds...")
    avg_thresh_super, avg_thresh_sub = calibrate_threshold_inner_loop(
        cfg, train_dataset, inner_seeds
    )
    if cfg["verbose"]:
        print(f"    Avg Threshold: Super={avg_thresh_super:.4f}, Sub={avg_thresh_sub:.4f}")
    
    # 3. 在 Train 上训练最终模型
    set_seed(outer_seed)
    
    result = run_training(
        feature_dim=cfg["feature_dim"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        device=device,
        enable_feature_gating=cfg["enable_feature_gating"],
        training_loss=cfg["training_loss"],
        train_features=train_dataset.features,
        train_super_labels=train_dataset.super_labels,
        train_sub_labels=train_dataset.sub_labels,
        output_dir=None,
        verbose=cfg["verbose"]
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
    
    # 4. 在 Test 上推理
    test_features = test_dataset.features.to(device)
    test_super_labels = test_dataset.super_labels
    test_sub_labels = test_dataset.sub_labels
    
    if cfg["enable_openmax"]:
        # OpenMax 分支：需要先拟合 Weibull
        if cfg["enable_feature_gating"]:
            # 拟合 OpenMax for super (使用全部训练数据)
            openmax_super = OpenMax(
                num_classes=len(super_map),
                weibull_tail_size=cfg["openmax_config"].weibull_tail_size,
                alpha=cfg["openmax_config"].alpha,
                distance_type=cfg["openmax_config"].distance_type
            )
            with torch.no_grad():
                train_super_logits, _ = model(train_dataset.features.to(device))
            train_super_labels_local = torch.tensor([super_map[l.item()] for l in train_dataset.super_labels])
            openmax_super.fit(train_super_logits, train_super_labels_local)
            
            # 拟合 OpenMax for sub
            openmax_sub = OpenMax(
                num_classes=len(sub_map),
                weibull_tail_size=cfg["openmax_config"].weibull_tail_size,
                alpha=cfg["openmax_config"].alpha,
                distance_type=cfg["openmax_config"].distance_type
            )
            with torch.no_grad():
                _, train_sub_logits = model(train_dataset.features.to(device))
            train_sub_labels_local = torch.tensor([sub_map[l.item()] for l in train_dataset.sub_labels])
            openmax_sub.fit(train_sub_logits, train_sub_labels_local)
            
            # OpenMax 推理
            super_preds, sub_preds, super_scores, sub_scores = predict_with_openmax(
                test_features, model, openmax_super, openmax_sub,
                super_map_inv, sub_map_inv,
                avg_thresh_super, avg_thresh_sub,
                cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
                super_to_sub
            )
        else:
            raise NotImplementedError("OpenMax 目前仅支持 enable_feature_gating=True 模式")
    elif cfg["enable_feature_gating"]:
        super_preds, sub_preds, super_scores, sub_scores = predict_with_gated_dual_head(
            test_features, model, super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
            super_to_sub, cfg["prediction_score_temperature"], cfg["prediction_score_method"]
        )
    else:
        super_preds, sub_preds, super_scores, sub_scores = predict_with_linear_single_head(
            test_features, super_model, sub_model,
            super_map_inv, sub_map_inv,
            avg_thresh_super, avg_thresh_sub, cfg["novel_super_idx"], cfg["novel_sub_idx"], device,
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
    super_binary_labels = (test_super_labels.numpy() != cfg["novel_super_idx"]).astype(int)
    sub_binary_labels = (test_sub_labels.numpy() != cfg["novel_sub_idx"]).astype(int)

    # OpenMax 返回的是 unknown 概率而不是 known 概率，需要取反
    if cfg["enable_openmax"]:
        super_scores_for_auroc = [1 - s for s in super_scores]  # 越高越 known
        sub_scores_for_auroc = [1 - s for s in sub_scores]
    else:
        super_scores_for_auroc = super_scores
        sub_scores_for_auroc = sub_scores

    if len(np.unique(super_binary_labels)) > 1:
        super_auroc = roc_auc_score(super_binary_labels, super_scores_for_auroc)
    else:
        super_auroc = float('nan')
    
    if len(np.unique(sub_binary_labels)) > 1:
        sub_auroc = roc_auc_score(sub_binary_labels, sub_scores_for_auroc)
    else:
        sub_auroc = float('nan')
    
    return {
        "super_overall": super_all, "super_seen": super_seen, "super_unseen": super_unseen,
        "sub_overall": sub_all, "sub_seen": sub_seen, "sub_unseen": sub_unseen,
        "super_auroc": super_auroc, "sub_auroc": sub_auroc
    }



def run_evaluation(cfg: dict, outer_seeds: list[int], inner_seeds: list[int]) -> dict:
    """
    运行完整评估，返回聚合统计结果
    """
    all_results = []
    
    for i, outer_seed in enumerate(outer_seeds):
        print(f"\n{'='*60}")
        print(f"Outer Trial {i+1}/{len(outer_seeds)} | Seed: {outer_seed}")
        print("="*60)
        result = run_single_outer_trial(cfg, outer_seed, inner_seeds)
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


def print_global_config(cfg: dict, outer_seeds: list[int], inner_seeds: list[int]):
    """打印全局配置"""
    mode = "SE Feature Gating" if cfg["enable_feature_gating"] else "Independent Training"
    masking = "Enabled" if cfg["enable_hierarchical_masking"] else "Disabled"
    
    print("=" * 60)
    print("Global Configuration (Nested Validation)")
    print("=" * 60)
    
    print("\n[Model]")
    print(f"  Mode: {mode}")
    print(f"  Hierarchical Masking: {masking}")
    
    print("\n[Training]")
    print(f"  Loss: {cfg['training_loss'].value.upper()}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch Size: {cfg['batch_size']}")
    print(f"  Learning Rate: {cfg['learning_rate']}")
    
    print("\n[Outer Split: Full -> Train + Test]")
    print(f"  Test Ratio: {cfg['test_ratio']*100:.0f}%")
    print(f"  Test Sub Novel Ratio: {cfg['test_sub_novel_ratio']*100:.0f}%")
    
    print("\n[Inner Split: Train -> SubTrain + Val]")
    print(f"  Val Ratio: {cfg['val_ratio']*100:.0f}%")
    print(f"  Val Sub Novel Ratio: {cfg['val_sub_novel_ratio']*100:.0f}%")
    print(f"  Val Include Novel: {cfg['val_include_novel']}")
    print(f"  Force Super Novel: {cfg['force_super_novel']}")
    
    print("\n[Threshold]")
    if cfg["enable_openmax"]:
        print(f"  Method: OpenMax + EER")
    elif cfg["val_include_novel"]:
        print(f"  Method: {cfg['full_val_threshold'].value} (val has unknown)")
    else:
        print(f"  Method: {cfg['known_only_threshold'].value} (val has no unknown)")

    print("\n[OOD Score]")
    if cfg["enable_openmax"]:
        print(f"  Method: OpenMax (Weibull tail={cfg['openmax_config'].weibull_tail_size}, alpha={cfg['openmax_config'].alpha})")
    else:
        print(f"  Validation: {cfg['validation_score_method'].value} (T={cfg['validation_score_temperature']})")
        print(f"  Prediction: {cfg['prediction_score_method'].value} (T={cfg['prediction_score_temperature']})")
    
    print("\n[Evaluation]")
    print(f"  Outer Seeds: {outer_seeds}")
    print(f"  Inner Seeds (for randomization): {inner_seeds}")
    print(f"  Total Outer Trials: {len(outer_seeds)}")
    print(f"  Inner Trials per Outer: Iterate all superclasses (3)")



if __name__ == "__main__":
    print_global_config(CONFIG, OUTER_SEEDS, INNER_SEEDS)
    
    stats = run_evaluation(CONFIG, OUTER_SEEDS, INNER_SEEDS)
    print_evaluation_report(stats)
