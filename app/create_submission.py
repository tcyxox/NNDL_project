import torch
import torch.nn.functional as F
import pandas as pd
import os
import json
from tqdm import tqdm
import torch.nn as nn

# ================= 配置 =================
CONFIG = {
    # 1. 训练好的模型和映射表所在位置 (来自 04_train_model.py)
    "model_dir": "baseline_models",

    # 2. 验证集位置 (用于自动计算最佳阈值, 来自 04_split_dataset_osr.py)
    "val_data_dir": "split_data_osr",

    # 3. 最终测试集特征位置 (来自 01_extract_features.py, 这是我们要预测的目标)
    "test_feature_path": "precomputed_features/test_features.pt",
    "test_image_names": "precomputed_features/test_image_names.pt",

    "feature_dim": 512,
    "output_csv": "submission_osr.csv",

    # 结果中的 Novel ID
    "novel_super_idx": 3,
    "novel_sub_idx": 87,

    # 阈值设定策略: 在验证集上保留多少比例的已知样本 (Recall)
    "target_recall": 0.95
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 模型定义 =================
class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


# ================= 核心函数：加载映射 =================
def load_mapping_and_model(prefix):
    """
    加载 json 映射表和对应的模型
    prefix: 'superclass' or 'subclass'
    """
    # 1. 加载映射表 (Local ID -> Global ID)
    json_path = os.path.join(CONFIG["model_dir"], f"{prefix}_mapping.json")
    with open(json_path, 'r') as f:
        # JSON key 默认是 str, 需要转回 int
        local_to_global = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(local_to_global)
    print(f"[{prefix}] 加载映射表: 检测到 {num_classes} 个已知类")

    # 2. 初始化模型 (使用动态的类别数量)
    model = LinearClassifier(CONFIG["feature_dim"], num_classes)
    model_path = os.path.join(CONFIG["model_dir"], f"{prefix}_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, local_to_global


# ================= 核心函数：自动计算阈值 =================
def calculate_threshold(model, val_features, val_labels, label_map, target_recall=0.95):
    """
    在验证集上运行，找出能覆盖 95% 已知样本的阈值
    """
    model.eval()

    # 筛选出验证集里的"已知类"样本 (标签在映射表里的)
    # 注意：val_labels 里有 87 (Novel)，我们要把它排除掉，只看已知类的分布
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    if len(X_known) == 0:
        print("警告: 验证集中没有已知类样本，使用默认阈值 0.5")
        return 0.5

    with torch.no_grad():
        logits = model(X_known)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)  # 获取每个已知样本对自己类别的置信度

    # 计算分位数: 找到一个分数 T，使得 95% 的样本分数都 > T
    threshold = torch.quantile(max_probs, 1 - target_recall).item()
    return threshold


# ================= 主程序 =================
if __name__ == "__main__":
    # --- 1. 加载模型和映射 ---
    print("--- 步骤 1: 加载模型和映射 ---")
    super_model, super_map = load_mapping_and_model("superclass")
    sub_model, sub_map = load_mapping_and_model("subclass")

    # --- 2. 利用验证集计算最佳阈值 ---
    print("\n--- 步骤 2: 计算最佳 OSR 阈值 ---")
    # 加载验证集 (从之前切分好的文件夹里)
    val_feat = torch.load(os.path.join(CONFIG["val_data_dir"], "val_features.pt"))
    val_super_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_super_labels.pt"))
    val_sub_lbl = torch.load(os.path.join(CONFIG["val_data_dir"], "val_sub_labels.pt"))

    thresh_super = calculate_threshold(super_model, val_feat, val_super_lbl, super_map, CONFIG["target_recall"])
    thresh_sub = calculate_threshold(sub_model, val_feat, val_sub_lbl, sub_map, CONFIG["target_recall"])

    print(f"  > 自动计算出的 Superclass 阈值: {thresh_super:.4f}")
    print(f"  > 自动计算出的 Subclass 阈值:   {thresh_sub:.4f}")

    # --- 3. 对官方测试集进行推理 ---
    print("\n--- 步骤 3: 生成最终预测 ---")
    test_features = torch.load(CONFIG["test_feature_path"]).to(device)
    test_image_names = torch.load(CONFIG["test_image_names"])

    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(test_features)), desc="Inference"):
            feature = test_features[i].unsqueeze(0)
            image_name = test_image_names[i]

            # === 超类预测 ===
            super_logits = super_model(feature)
            super_probs = F.softmax(super_logits, dim=1)
            max_s_prob, s_idx = torch.max(super_probs, dim=1)

            # 阈值判断
            if max_s_prob.item() < thresh_super:
                final_super = CONFIG["novel_super_idx"]
            else:
                # [关键] 映射回原始 ID
                local_id = s_idx.item()
                final_super = super_map[local_id]

            # === 子类预测 ===
            sub_logits = sub_model(feature)
            sub_probs = F.softmax(sub_logits, dim=1)
            max_sub_prob, sub_idx = torch.max(sub_probs, dim=1)

            # 阈值判断
            if max_sub_prob.item() < thresh_sub:
                final_sub = CONFIG["novel_sub_idx"]
            else:
                # [关键] 映射回原始 ID
                local_id = sub_idx.item()
                final_sub = sub_map[local_id]

            # === 逻辑一致性修正 ===
            # 如果超类是 novel，子类必须是 novel
            if final_super == CONFIG["novel_super_idx"]:
                final_sub = CONFIG["novel_sub_idx"]

            predictions.append({
                "image": image_name,
                "superclass_index": final_super,
                "subclass_index": final_sub
            })

    # --- 4. 保存 ---
    df = pd.DataFrame(predictions)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"提交文件已保存至: {CONFIG['output_csv']}")