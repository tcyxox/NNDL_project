import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json

from config import *
from utils import LinearClassifier

CONFIG = {
    "feature_dir": SPLIT_DIR,
    "output_dir": MODELS_DIR,
    "feature_dim": FEATURE_DIM,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================== 加载划分后的训练数据 ===============================
print("正在加载划分后的训练数据 (Train Split)...")
# 注意：只加载 train_*.pt，因为只有已知类才能用于训练
train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))


# =============================== 创建标签映射 ===============================
def create_label_mapping(labels, label_name):
    """
    因为训练集中可能缺少某些类（作为未知类），导致标签不连续（如 0, 1, 5...）。
    我们需要将其映射为连续的 ID (0, 1, 2...) 才能输入 CrossEntropyLoss。
    """
    unique_classes = torch.unique(labels).sort()[0].tolist()
    num_classes = len(unique_classes)

    # 映射字典: 原始ID -> 模型内部ID (用于训练)
    global_to_local = {original: local for local, original in enumerate(unique_classes)}
    # 反向字典: 模型内部ID -> 原始ID (用于推理恢复)
    local_to_global = {local: original for local, original in enumerate(unique_classes)}

    print(f"[{label_name}] 检测到 {num_classes} 个已知类别。")
    print(f"  > 原始标签示例: {unique_classes[:5]}...")

    # 保存映射关系，推理时必须用！
    mapping_path = os.path.join(CONFIG["output_dir"], f"{label_name}_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(local_to_global, f)  # 保存反向映射，方便最后生成提交文件
    print(f"  > 映射表已保存至: {mapping_path}")

    return num_classes, global_to_local


# 1. 处理超类 (虽然通常超类都已知，但为了程序健壮性，统一处理)
num_super, super_map = create_label_mapping(train_super_labels, "superclass")

# 2. 处理子类 (核心：将过滤掉 Novel 后的剩余子类重新编号)
num_sub, sub_map = create_label_mapping(train_sub_labels, "subclass")


# =============================== 训练模型 ===============================
def train_model(features, labels, label_map, num_classes, model_name):
    # 初始化模型，输出维度 = 已知类的数量
    model = LinearClassifier(CONFIG["feature_dim"], num_classes)
    model.to(device)
    model.train()

    # 将所有标签转换为 Local ID
    # 这一步至关重要：把原来的 [0, 5, 12] 变成 [0, 1, 2]
    mapped_labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)

    # 创建数据集
    dataset = TensorDataset(features, mapped_labels)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"\n开始训练 {model_name} (Classes: {num_classes})...")

    for epoch in range(CONFIG["epochs"]):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{CONFIG['epochs']}], Loss: {running_loss / len(loader):.4f}")

    return model


# =============================== 执行训练 ===============================

# 1. 训练超类模型
super_model = train_model(train_features, train_super_labels, super_map, num_super, "Superclass Model")
torch.save(super_model.state_dict(), os.path.join(CONFIG["output_dir"], "superclass_model.pth"))
print("超类模型已保存。")

# 2. 训练子类模型
sub_model = train_model(train_features, train_sub_labels, sub_map, num_sub, "Subclass Model")
torch.save(sub_model.state_dict(), os.path.join(CONFIG["output_dir"], "subclass_model.pth"))
print("子类模型已保存。")

print("\n--- 所有模型训练完毕 ---")
print(f"请检查 {CONFIG['output_dir']} 目录下的模型文件 (.pth) 和 映射文件 (.json)。")