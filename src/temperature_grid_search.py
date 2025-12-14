import json
import os
import time
from datetime import datetime

import numpy as np
import sys

from core.config import config, TrainingLoss, OODScoreMethod
from evaluate_performance import run_multiple_trials, print_evaluation_report

# 温度值列表
TEMPERATURES = [0.02, 0.1, 0.2, 0.5, 1, 1.2, 1.5, 2, 3, 3.5, 4]

# 多种子评估配置
SEEDS = [42, 123, 456, 789, 1024]

# 基础配置
BASE_CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": config.paths.dev,
    "feature_dim": config.model.feature_dim,
    "learning_rate": config.experiment.learning_rate,
    "batch_size": config.experiment.batch_size,
    "epochs": config.experiment.epochs,
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    "target_recall": config.experiment.target_recall,
    # 固定配置
    "enable_feature_gating": True,
    "enable_hierarchical_masking": True,
    "training_loss": TrainingLoss.BCE,
    "threshold_method": OODScoreMethod.Energy,
    "prediction_method": OODScoreMethod.Energy,
}

def run_grid_search():
    """执行温度参数网格搜索"""
    results = []
    total_combinations = len(TEMPERATURES) ** 2
    current = 0
    
    # 输出目录
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 结果文件路径
    results_file = os.path.join(outputs_dir, "temperature_grid_search_results.json")
    
    start_time = time.time()
    
    print("=" * 80)
    print("Temperature Grid Search")
    print(f"Temperature values: {TEMPERATURES}")
    print(f"Total combinations: {total_combinations}")
    print(f"Seeds per combination: {len(SEEDS)}")
    print("=" * 80)
    
    for thresh_temp in TEMPERATURES:
        for pred_temp in TEMPERATURES:
            current += 1
            
            print(f"\n{'='*80}")
            print(f"[{current}/{total_combinations}] threshold_temp={thresh_temp}, prediction_temp={pred_temp}")
            print(f"{'='*80}")
            
            # 创建配置副本并更新温度参数
            cfg = BASE_CONFIG.copy()
            cfg["threshold_temperature"] = thresh_temp
            cfg["prediction_temperature"] = pred_temp
            
            try:
                # 运行多种子评估
                stats = run_multiple_trials(cfg, SEEDS, verbose=False)
                
                # 保存结果
                result = {
                    "threshold_temperature": thresh_temp,
                    "prediction_temperature": pred_temp,
                    "super_overall_mean": float(stats["super_overall_mean"]),
                    "super_overall_std": float(stats["super_overall_std"]),
                    "super_seen_mean": float(stats["super_seen_mean"]),
                    "super_seen_std": float(stats["super_seen_std"]),
                    "super_unseen_mean": float(stats["super_unseen_mean"]),
                    "super_unseen_std": float(stats["super_unseen_std"]),
                    "sub_overall_mean": float(stats["sub_overall_mean"]),
                    "sub_overall_std": float(stats["sub_overall_std"]),
                    "sub_seen_mean": float(stats["sub_seen_mean"]),
                    "sub_seen_std": float(stats["sub_seen_std"]),
                    "sub_unseen_mean": float(stats["sub_unseen_mean"]),
                    "sub_unseen_std": float(stats["sub_unseen_std"]),
                    "super_auroc_mean": float(stats["super_auroc_mean"]) if not np.isnan(stats["super_auroc_mean"]) else None,
                    "super_auroc_std": float(stats["super_auroc_std"]) if not np.isnan(stats["super_auroc_std"]) else None,
                    "sub_auroc_mean": float(stats["sub_auroc_mean"]) if not np.isnan(stats["sub_auroc_mean"]) else None,
                    "sub_auroc_std": float(stats["sub_auroc_std"]) if not np.isnan(stats["sub_auroc_std"]) else None,
                }
                
                results.append(result)
                
                # 打印当前结果
                print(f"\n  [Subclass] Overall: {result['sub_overall_mean']*100:.2f}% ± {result['sub_overall_std']*100:.2f}%")
                print(f"  [Subclass] Seen   : {result['sub_seen_mean']*100:.2f}% ± {result['sub_seen_std']*100:.2f}%")
                print(f"  [Subclass] Unseen : {result['sub_unseen_mean']*100:.2f}% ± {result['sub_unseen_std']*100:.2f}%")
                print(f"  [Subclass] AUROC  : {result['sub_auroc_mean']:.4f} ± {result['sub_auroc_std']:.4f}")
                
                # 每次完成后保存中间结果
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "config": {
                            "temperatures": TEMPERATURES,
                            "seeds": SEEDS,
                            "enable_feature_gating": True,
                            "enable_hierarchical_masking": True,
                            "training_loss": "CE",
                            "threshold_method": "MSP",
                            "prediction_method": "MSP",
                        },
                        "results": results,
                        "completed": current,
                        "total": total_combinations,
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"  Results saved to: {results_file}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                # 记录错误结果
                result = {
                    "threshold_temperature": thresh_temp,
                    "prediction_temperature": pred_temp,
                    "error": str(e),
                }
                results.append(result)
    
    # 计算总耗时
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n{'='*80}")
    print(f"Grid Search Complete!")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    return results


def analyze_results(results_file: str = None):
    """分析网格搜索结果"""
    if results_file is None:
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        results_file = os.path.join(outputs_dir, "temperature_grid_search_results.json")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data["results"]
    
    print("\n" + "=" * 80)
    print("Temperature Grid Search Analysis")
    print("=" * 80)
    
    # 过滤掉错误结果
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("No valid results found!")
        return
    
    # 按 Subclass Unseen 准确率排序
    sorted_by_unseen = sorted(valid_results, key=lambda x: x["sub_unseen_mean"], reverse=True)
    
    print("\n--- Top 10 by Subclass Unseen Accuracy ---")
    print(f"{'Thresh T':<10} {'Pred T':<10} {'Unseen':<15} {'Overall':<15} {'Seen':<15} {'AUROC':<15}")
    print("-" * 80)
    for r in sorted_by_unseen[:10]:
        print(f"{r['threshold_temperature']:<10} {r['prediction_temperature']:<10} "
              f"{r['sub_unseen_mean']*100:.2f}%±{r['sub_unseen_std']*100:.2f}% "
              f"{r['sub_overall_mean']*100:.2f}%±{r['sub_overall_std']*100:.2f}% "
              f"{r['sub_seen_mean']*100:.2f}%±{r['sub_seen_std']*100:.2f}% "
              f"{r['sub_auroc_mean']:.4f}")
    
    # 按 Subclass Overall 准确率排序
    sorted_by_overall = sorted(valid_results, key=lambda x: x["sub_overall_mean"], reverse=True)
    
    print("\n--- Top 10 by Subclass Overall Accuracy ---")
    print(f"{'Thresh T':<10} {'Pred T':<10} {'Overall':<15} {'Unseen':<15} {'Seen':<15} {'AUROC':<15}")
    print("-" * 80)
    for r in sorted_by_overall[:10]:
        print(f"{r['threshold_temperature']:<10} {r['prediction_temperature']:<10} "
              f"{r['sub_overall_mean']*100:.2f}%±{r['sub_overall_std']*100:.2f}% "
              f"{r['sub_unseen_mean']*100:.2f}%±{r['sub_unseen_std']*100:.2f}% "
              f"{r['sub_seen_mean']*100:.2f}%±{r['sub_seen_std']*100:.2f}% "
              f"{r['sub_auroc_mean']:.4f}")
    
    # 按 AUROC 排序
    sorted_by_auroc = sorted(valid_results, key=lambda x: x["sub_auroc_mean"] if x["sub_auroc_mean"] else 0, reverse=True)
    
    print("\n--- Top 10 by Subclass AUROC ---")
    print(f"{'Thresh T':<10} {'Pred T':<10} {'AUROC':<15} {'Overall':<15} {'Unseen':<15} {'Seen':<15}")
    print("-" * 80)
    for r in sorted_by_auroc[:10]:
        print(f"{r['threshold_temperature']:<10} {r['prediction_temperature']:<10} "
              f"{r['sub_auroc_mean']:.4f}±{r['sub_auroc_std']:.4f} "
              f"{r['sub_overall_mean']*100:.2f}%±{r['sub_overall_std']*100:.2f}% "
              f"{r['sub_unseen_mean']*100:.2f}%±{r['sub_unseen_std']*100:.2f}% "
              f"{r['sub_seen_mean']*100:.2f}%±{r['sub_seen_std']*100:.2f}%")
    
    # 最佳配置
    best_unseen = sorted_by_unseen[0]
    best_overall = sorted_by_overall[0]
    best_auroc = sorted_by_auroc[0]
    
    print("\n" + "=" * 80)
    print("Best Configurations Summary")
    print("=" * 80)
    print(f"\nBest for Subclass Unseen:")
    print(f"  threshold_temperature: {best_unseen['threshold_temperature']}")
    print(f"  prediction_temperature: {best_unseen['prediction_temperature']}")
    print(f"  Unseen: {best_unseen['sub_unseen_mean']*100:.2f}% ± {best_unseen['sub_unseen_std']*100:.2f}%")
    
    print(f"\nBest for Subclass Overall:")
    print(f"  threshold_temperature: {best_overall['threshold_temperature']}")
    print(f"  prediction_temperature: {best_overall['prediction_temperature']}")
    print(f"  Overall: {best_overall['sub_overall_mean']*100:.2f}% ± {best_overall['sub_overall_std']*100:.2f}%")
    
    print(f"\nBest for Subclass AUROC:")
    print(f"  threshold_temperature: {best_auroc['threshold_temperature']}")
    print(f"  prediction_temperature: {best_auroc['prediction_temperature']}")
    print(f"  AUROC: {best_auroc['sub_auroc_mean']:.4f} ± {best_auroc['sub_auroc_std']:.4f}")


if __name__ == "__main__":
    analyze = True
    
    if analyze or (len(sys.argv) > 1 and sys.argv[1] == "--analyze"):
        # 只分析已有结果
        results_file = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_results(results_file)
    else:
        # 运行网格搜索
        results = run_grid_search()
        # 分析结果
        analyze_results()
