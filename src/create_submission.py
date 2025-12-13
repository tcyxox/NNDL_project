import json
import os

import pandas as pd
import torch

from core.config import config
from core.inference import load_linear_model, predict_with_linear_model, load_hierarchical_model, \
    predict_with_hierarchical_model

CONFIG = {
    "hyperparams_file": os.path.join(config.paths.dev, "hyperparameters.json"),
    "model_dir": config.paths.submit,
    "test_feature_path": os.path.join(config.paths.features, "test_features.pt"),
    "test_image_names": os.path.join(config.paths.features, "test_image_names.pt"),
    "output_csv": os.path.join(config.paths.outputs, "submission_osr.csv"),
    "novel_super_idx": config.osr.novel_super_index,
    "novel_sub_idx": config.osr.novel_sub_index,
    "enable_hierarchical_masking": config.experiment.enable_hierarchical_masking,
    "feature_dim": config.model.feature_dim,
    "enable_feature_gating": config.experiment.enable_feature_gating,
    "enable_energy": config.experiment.enable_energy,
    "enable_sigmoid_bce": config.experiment.enable_sigmoid_bce,
    "ood_temperature": config.experiment.ood_temperature,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # --- Step 1: 加载模型和映射 ---
    print("--- Step 1: 加载模型和映射 ---")
    
    # 加载映射表 (获取类别数量)
    with open(os.path.join(CONFIG["model_dir"], "super_local_to_global_map.json"), 'r') as f:
        super_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(CONFIG["model_dir"], "sub_local_to_global_map.json"), 'r') as f:
        sub_map = {int(k): v for k, v in json.load(f).items()}
    
    num_super, num_sub = len(super_map), len(sub_map)
    
    # 加载超类到子类的映射表（用于 hierarchical masking）
    super_to_sub = None
    if CONFIG["enable_hierarchical_masking"]:
        with open(os.path.join(CONFIG["model_dir"], "super_to_sub_map.json"), 'r') as f:
            super_to_sub = {int(k): v for k, v in json.load(f).items()}
        print(f"  > Hierarchical masking 已启用")
    else:
        print(f"  > Hierarchical masking 已禁用")

    # --- Step 2: 加载阈值 ---
    print("\n--- Step 2: 加载阈值 ---")
    with open(CONFIG["hyperparams_file"], 'r') as f:
        hyperparams = json.load(f)
    thresh_super = hyperparams["thresh_super"]
    thresh_sub = hyperparams["thresh_sub"]
    print(f"  > Superclass 阈值: {thresh_super:.4f}")
    print(f"  > Subclass 阈值:   {thresh_sub:.4f}")

    # --- Step 3: 测试集推理 ---
    print("\n--- Step 3: 测试集推理 ---")
    test_features = torch.load(CONFIG["test_feature_path"]).to(device)
    test_image_names = torch.load(CONFIG["test_image_names"])
    
    use_energy = CONFIG["enable_energy"]
    use_sigmoid_bce = CONFIG["enable_sigmoid_bce"]

    if CONFIG["enable_feature_gating"]:
        print("  > 使用 Soft Attention 模式")
        model, super_map, sub_map = load_hierarchical_model(
            CONFIG["model_dir"], CONFIG["feature_dim"], num_super, num_sub, True, device
        )
        super_preds, sub_preds, _, _ = predict_with_hierarchical_model(
            test_features, model, super_map, sub_map,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, use_energy, temperature=CONFIG["ood_temperature"],
            use_sigmoid_bce=use_sigmoid_bce
        )
    else:
        print("  > 使用独立模型模式")
        super_model, super_map = load_linear_model("super", CONFIG["model_dir"], CONFIG["feature_dim"], device)
        sub_model, sub_map = load_linear_model("sub", CONFIG["model_dir"], CONFIG["feature_dim"], device)
        super_preds, sub_preds, _, _ = predict_with_linear_model(
            test_features, super_model, sub_model,
            super_map, sub_map,
            thresh_super, thresh_sub,
            CONFIG["novel_super_idx"], CONFIG["novel_sub_idx"], device,
            super_to_sub, CONFIG["enable_energy"], temperature=CONFIG["ood_temperature"]
        )

    # --- Step 4: 保存提交文件 ---
    print("\n--- Step 4: 保存提交文件 ---")
    predictions = [
        {"image": test_image_names[i], "superclass_index": super_preds[i], "subclass_index": sub_preds[i]}
        for i in range(len(test_image_names))
    ]
    df = pd.DataFrame(predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"  > 提交文件已保存至: {CONFIG['output_csv']}")
