import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

from src.OpenMax.OpenMax import *
from src.core.utils import set_seed
from src.core.config import *
from src.core.train import create_label_mapping


CONFIG = {
    "feature_dir": config.paths.split_features,
    "output_dir": os.path.join(config.paths.dev, "OpenMax"),
    "feature_dim": config.model.feature_dim,
    "learning_rate": 0.001,
    "batch_size": config.experiment.batch_size,
    "epochs": 10,
    "alpha": 10,
    "weibull_tail_size": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

def train_openmax_classifier(
        train_features,
        train_labels,
        val_features,
        val_labels,
        label_map,
        model_name="OpenMax_Model",
        feature_dim=512,
        num_classes=70,  # 已知类数量
        alpha=10,
        weibull_tail_size=20,
        distance_type="cosine",
        output_dir=None,
        seed=42,
        batch_size=32,
        epochs=50,
        lr=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"[{model_name}] 初始化设置: Alpha={alpha}, Tail={weibull_tail_size}")
    set_seed(seed)

    linear_clf = LinearClassifier(feature_dim, num_classes)
    linear_clf.to(device)

    # 训练集映射
    mapped_train_labels = torch.tensor([label_map[l.item()] for l in train_labels], dtype=torch.long)
    train_dataset = TensorDataset(train_features, mapped_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 验证集映射
    mapped_val_labels = torch.tensor([label_map[l.item()] for l in val_labels], dtype=torch.long)
    val_dataset = TensorDataset(val_features, mapped_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 损失与优化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_clf.parameters(), lr=lr)

    print(f"开始训练 {model_name} - 线性分类器")

    linear_clf.train()
    for epoch in range(epochs):
        # =============== 训练阶段 ===============
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = linear_clf(batch_X)  # 获取 Logits
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # =============== 验证阶段 ===============
        linear_clf.eval()
        val_loss = 0.0
        seen_correct = 0
        total_seen = 0

        # 存储 AUROC 计算所需的数据
        all_scores = []
        all_binary_labels = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = linear_clf(inputs)  # 获取 Logits

                mask = targets != -1

                # ====== 对所有类别计算 AUROC 分数 ======
                probs = F.softmax(outputs, dim=1)
                # 取最大概率作为"属于已知类"的置信度
                msp_scores, _ = torch.max(probs, dim=1)
                msp_scores = msp_scores.cpu().numpy()
                all_scores.extend(msp_scores)
                is_known = mask.cpu().int().tolist()
                all_binary_labels.extend(is_known)

                # ====== 对已知类计算 Loss 和 Accuracy ======
                if mask.sum() > 0:
                    # 筛选出已知类
                    known_targets = targets[mask]
                    known_outputs = outputs[mask]
                    # loss
                    loss = criterion(known_outputs, known_targets)
                    val_loss += loss.item()
                    # accuracy
                    _, known_predicted = torch.max(known_outputs, 1)
                    total_seen += known_targets.size(0)
                    seen_correct += (known_predicted == known_targets).sum().item()

        # 计算各指标
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * seen_correct / total_seen if total_seen > 0 else 0.0

        val_auroc = 0.5
        if len(np.unique(all_binary_labels)) == 2:
            val_auroc = roc_auc_score(all_binary_labels, -np.array(all_scores))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Seen Acc: {val_acc:.2f}% | "
                  f"Val AUROC: {val_auroc:.4f}")

    # ===================== 校准 OpenMax (Fit) =====================
    print("校准 OpenMax...")
    linear_clf.eval()
    all_logits = []
    all_gt = []

    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            logits = linear_clf(batch_X)
            all_logits.append(logits.cpu())
            all_gt.append(batch_y.cpu())

    all_logits = torch.cat(all_logits)
    all_gt = torch.cat(all_gt)

    # OpenMax 只使用预测正确的样本进行校准
    preds = torch.argmax(all_logits, dim=1)
    correct_mask = preds == all_gt

    correct_logits = all_logits[correct_mask]
    correct_labels = all_gt[correct_mask]

    print(f"  - 使用 {len(correct_labels) / len(all_gt) * 100:.4f} % 正确已知类样本进行 Weibull 拟合")

    # 初始化并拟合 OpenMax
    openmax = OpenMax(
        num_classes=num_classes,
        weibul_tail_size=weibull_tail_size,
        alpha=alpha,
        distance_type=distance_type
    )

    openmax.fit(correct_logits, correct_labels)

    system = OpenMaxSystem(linear_clf, openmax, device)
    # 保存模型
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(linear_clf.state_dict(), os.path.join(output_dir, f"{model_name}_linear.pth"))
        # 保存OpenMax
        with open(os.path.join(output_dir, f"best_{model_name}_seed_{seed}.pkl"), "wb") as f:
            pickle.dump(openmax, f)
        # 保存线性分类器
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f"best_{model_name}_seed_{seed}.pth"))
        print(f"  - 模型已保存至 {output_dir}")

    return system


if __name__ == "__main__":
    # =============================== 加载划分后的训练数据 ===============================
    print("加载训练集 (Train Split)...")
    train_features = torch.load(os.path.join(CONFIG["feature_dir"], "train_features.pt"))
    train_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_super_labels.pt"))
    train_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "train_sub_labels.pt"))
    print("加载验证集 (Val Split)...")
    val_features = torch.load(os.path.join(CONFIG["feature_dir"], "val_features.pt"))
    val_super_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_super_labels.pt"))
    val_sub_labels = torch.load(os.path.join(CONFIG["feature_dir"], "val_sub_labels.pt"))

    # =============================== 创建标签映射 ===============================
    num_super, super_map = create_label_mapping(train_super_labels, "super", CONFIG["output_dir"])
    num_sub, sub_map = create_label_mapping(train_sub_labels, "sub", CONFIG["output_dir"])
    sub_map[87] = -1

    # 开始训练
    model = train_openmax_classifier(
        train_features=train_features,
        train_labels=train_sub_labels,
        val_features=val_features,
        val_labels=val_sub_labels,
        label_map=sub_map,
        num_classes=num_sub,
        model_name="OpenMax_Model",
        feature_dim=CONFIG["feature_dim"],
        alpha=CONFIG["alpha"],
        weibull_tail_size=CONFIG["weibull_tail_size"],
        output_dir=None,
        seed=42,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["learning_rate"],
    )