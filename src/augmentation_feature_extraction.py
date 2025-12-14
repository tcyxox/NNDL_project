import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
import numpy as np
import cv2  # 需要安装 opencv-python
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from core.config import config
from core.utils import set_seed

set_seed(config.experiment.seed)

CONFIG = {
    "train_csv_path": os.path.join(config.paths.data_raw, "train_data.csv"),
    "train_img_dir": os.path.join(config.paths.data_raw, "train_images"),
    "test_img_dir": os.path.join(config.paths.data_raw, "test_images"),
    "output_dir": os.path.join(config.paths.features, "augmentation_features"),
    "model_id": config.model.clip_model_id,
    "batch_size": config.experiment.batch_size,
    "unknown_label": -1,  # 定义未知类的标签索引，或者设为 class_num
    "use_opencutout": True  # 是否使用 OpenCutout (GrabCut速度较慢，默认关闭)
}

# 确保输出目录存在
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================== 加载模型 ===============================
print("正在加载 CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(CONFIG["model_id"]).to(device)
processor = CLIPProcessor.from_pretrained(CONFIG["model_id"])


# =============================== OpenMix 增强函数 ===============================
def open_mixup(pil_img, alpha=0.5):
    """
    OpenMixup: 原始图像与旋转后的自身混合
    """
    # 随机旋转
    # angle = random.uniform(0, 360)
    angle = random.uniform(45, 360-45)
    img_rotated = pil_img.rotate(angle)

    # 线性混合: x_new = 0.5 * x + 0.5 * x_rot
    # PIL blend 实现: out = image1 * (1.0 - alpha) + image2 * alpha
    mixed_img = Image.blend(pil_img, img_rotated, alpha=alpha)
    return mixed_img


def open_cutout(pil_img):
    """
    OpenCutout: 利用 GrabCut 移除前景，只保留背景
    注意：这步比较慢，仅在追求极致性能时开启
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = np.zeros(cv_img.shape[:2], np.uint8)

    # 定义矩形框 (假设物体在中心，留出边缘作为背景)
    h, w = cv_img.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        # 运行 GrabCut，迭代次数设为 1 或 2 以节省时间
        cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)

        # mask 中 0 和 2 是背景，1 和 3 是前景
        # 我们需要保留背景 (0, 2)，移除前景
        # OpenCutout 逻辑：保留背景，前景置黑
        mask_bg = np.where((mask == 0) | (mask == 2), 1, 0).astype('uint8')
        img_bg = cv_img * mask_bg[:, :, np.newaxis]

        return Image.fromarray(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
    except:
        # 如果分割失败，降级返回 OpenMixup
        return open_mixup(pil_img)


def generate_openmix_samples(image_paths, img_dir):
    """
    为当前批次生成对应的 OpenMix 负样本
    """
    mixed_images = []
    for img_path in image_paths:
        full_path = os.path.join(img_dir, img_path)
        original_img = Image.open(full_path).convert("RGB")

        # 策略选择：论文中结合了多种，这里为了效率可以随机选一种或者只用 Mixup
        # OpenMixup 速度最快且效果最显著
        if CONFIG["use_opencutout"] and random.random() < 0.7:
            neg_img = open_cutout(original_img)
        else:
            neg_img = open_mixup(original_img)

        mixed_images.append(neg_img)
    return mixed_images


# =============================== 辅助函数：处理单个批次 ===============================
@torch.no_grad()
def extract_features_from_pil(pil_images):
    """
    直接接受 PIL Image 列表，不再重复读取 IO
    """
    inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
    features = model.get_image_features(**inputs)
    return features.cpu()


@torch.no_grad()
def extract_text_features(text_list):
    inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features.cpu()


if __name__ == "__main__":
    # =============================== 处理训练数据 ===============================
    print("正在处理训练集...")
    train_df = pd.read_csv(CONFIG["train_csv_path"])
    train_images = train_df["image"].tolist()
    train_descriptions = train_df["description"].tolist()

    # 原始标签
    train_super_labels = torch.tensor(train_df["superclass_index"].values, dtype=torch.long)
    train_sub_labels = torch.tensor(train_df["subclass_index"].values, dtype=torch.long)

    # 存储列表
    train_features_list = []
    train_text_features_list = []
    openmix_features_list = []  # 存储 OpenMix 负样本特征

    for i in tqdm(range(0, len(train_images), CONFIG["batch_size"]), desc="提取训练特征"):
        batch_paths = train_images[i:i + CONFIG["batch_size"]]

        # 1. 读取原始图片并转换为 PIL 对象
        batch_pil_images = []
        for p in batch_paths:
            batch_pil_images.append(Image.open(os.path.join(CONFIG["train_img_dir"], p)).convert("RGB"))

        # 2. 提取原始图像特征 (Known Classes)
        batch_features = extract_features_from_pil(batch_pil_images)
        train_features_list.append(batch_features)

        # 3. 生成并提取 OpenMix 负样本特征 (Unknown Classes)
        batch_openmix_imgs = []
        for img in batch_pil_images:
            if CONFIG["use_opencutout"] and random.random() < 0.3:
                batch_openmix_imgs.append(open_cutout(img))
            else:
                batch_openmix_imgs.append(open_mixup(img))

        batch_openmix_features = extract_features_from_pil(batch_openmix_imgs)
        openmix_features_list.append(batch_openmix_features)

        # 4. 提取文本特征
        batch_texts = train_descriptions[i:i + CONFIG["batch_size"]]
        batch_text_feats = extract_text_features(batch_texts)
        train_text_features_list.append(batch_text_feats)

    # 合并张量
    train_features = torch.cat(train_features_list, dim=0)
    train_text_features = torch.cat(train_text_features_list, dim=0)
    openmix_features = torch.cat(openmix_features_list, dim=0)

    # ================= 保存 =================
    # 选项 A: 将 OpenMix 特征单独保存 (推荐，因为这样你可以灵活控制训练时混入多少负样本)
    # 选项 B: 直接拼接到 train_features 里

    # 这里采用选项 A，单独保存
    torch.save(train_features, os.path.join(CONFIG["output_dir"], "train_features.pt"))
    torch.save(openmix_features, os.path.join(CONFIG["output_dir"], "train_openmix_features.pt"))

    torch.save(train_text_features, os.path.join(CONFIG["output_dir"], "train_text_features.pt"))
    torch.save(train_super_labels, os.path.join(CONFIG["output_dir"], "train_super_labels.pt"))
    torch.save(train_sub_labels, os.path.join(CONFIG["output_dir"], "train_sub_labels.pt"))

    print(f"原始训练特征: {train_features.shape}")
    print(f"OpenMix负样本特征: {openmix_features.shape}")
    print(f"训练文本特征: {train_text_features.shape}")

    # =============================== 处理测试数据 ===============================
    print("正在处理测试集...")
    test_images = sorted([f for f in os.listdir(CONFIG["test_img_dir"]) if f.endswith('.jpg')])

    test_features_list = []
    # 注意：这里需要稍微改一下调用方式，因为我们上面修改了 extract_features 逻辑
    for i in tqdm(range(0, len(test_images), CONFIG["batch_size"]), desc="提取测试特征"):
        batch_paths = test_images[i:i + CONFIG["batch_size"]]
        batch_pil_images = [Image.open(os.path.join(CONFIG["test_img_dir"], p)).convert("RGB") for p in batch_paths]
        batch_features = extract_features_from_pil(batch_pil_images)
        test_features_list.append(batch_features)

    test_features = torch.cat(test_features_list, dim=0)

    torch.save(test_features, os.path.join(CONFIG["output_dir"], "test_features.pt"))
    torch.save(test_images, os.path.join(CONFIG["output_dir"], "test_image_names.pt"))
    print(f"测试特征已保存: {test_features.shape}")

    print("--- 特征提取完成 ---")