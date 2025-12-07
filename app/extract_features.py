import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from tqdm import tqdm

CONFIG = {
    "model_id": "openai/clip-vit-base-patch32",
    "train_csv_path": "data/train_data.csv",
    "test_csv_path": "data/example_test_predictions.csv",  # 获取测试图片列表
    "train_img_dir": "data/train_images",
    "test_img_dir": "data/test_images",
    "output_dir": "precomputed_features",  # 输出目录
    "batch_size": 64
}

# 确保输出目录存在
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================== 加载模型 ===============================
print("正在加载 CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = CLIPModel.from_pretrained(CONFIG["model_id"]).to(device)
processor = CLIPProcessor.from_pretrained(CONFIG["model_id"])


# =============================== 辅助函数：处理单个批次 ===============================
@torch.no_grad()
def extract_features(image_paths, img_dir):
    images = []
    for img_path in image_paths:
        try:
            full_path = os.path.join(img_dir, img_path)
            images.append(Image.open(full_path))
        except FileNotFoundError:
            print(f"警告: 找不到图片 {full_path}")
            images.append(Image.new('RGB', (224, 224), color='black'))  # 添加一个空图像占位

    # 预处理
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    # 提取图像特征
    features = model.get_image_features(**inputs)
    return features.cpu()


if __name__ == "__main__":
    # =============================== 处理训练数据 ===============================
    print("正在处理训练集...")
    train_df = pd.read_csv(CONFIG["train_csv_path"])
    train_images = train_df["image"].tolist()
    train_super_labels = torch.tensor(train_df["superclass_index"].values, dtype=torch.long)
    train_sub_labels = torch.tensor(train_df["subclass_index"].values, dtype=torch.long)

    train_features_list = []
    for i in tqdm(range(0, len(train_images), CONFIG["batch_size"]), desc="提取训练特征"):
        batch_paths = train_images[i:i + CONFIG["batch_size"]]
        batch_features = extract_features(batch_paths, CONFIG["train_img_dir"])
        train_features_list.append(batch_features)

    train_features = torch.cat(train_features_list, dim=0)

    # 保存训练特征和标签
    torch.save(train_features, os.path.join(CONFIG["output_dir"], "train_features.pt"))
    torch.save(train_super_labels, os.path.join(CONFIG["output_dir"], "train_super_labels.pt"))
    torch.save(train_sub_labels, os.path.join(CONFIG["output_dir"], "train_sub_labels.pt"))
    print(f"训练特征已保存: {train_features.shape}")

    # --- 处理测试数据 ---
    print("正在处理测试集...")
    test_df = pd.read_csv(CONFIG["test_csv_path"])
    test_images = test_df["image"].tolist()

    test_features_list = []
    for i in tqdm(range(0, len(test_images), CONFIG["batch_size"]), desc="提取测试特征"):
        batch_paths = test_images[i:i + CONFIG["batch_size"]]
        batch_features = extract_features(batch_paths, CONFIG["test_img_dir"])
        test_features_list.append(batch_features)

    test_features = torch.cat(test_features_list, dim=0)

    # 保存测试特征和图像名称
    torch.save(test_features, os.path.join(CONFIG["output_dir"], "test_features.pt"))
    torch.save(test_images, os.path.join(CONFIG["output_dir"], "test_image_names.pt"))  # 保存文件名用于最后提交
    print(f"测试特征已保存: {test_features.shape}")

    print("--- 特征提取完成 ---")