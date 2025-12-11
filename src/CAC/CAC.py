import torch
import torch.nn as nn
import torch.nn.functional as F


class CACProjector(nn.Module):
    """
    CAC (Class Anchor Clustering) 投影头
    用于将 CLIP 特征映射到符合 CAC 几何约束的 Logit 空间。
    """
    def __init__(self, input_dim, num_classes, alpha=10.0, se_reduction=-1, anchor_mode=None):
        super(CACProjector, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.se_reduction = se_reduction

        # --- SE Attention Module ---
        if self.se_reduction > 0:
            # CLIP 特征已经是 Global Token，所以跳过 Pooling，直接做 Channel Attention
            self.se_block = nn.Sequential(
                nn.Linear(input_dim, input_dim // self.se_reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // self.se_reduction, input_dim, bias=False),
                nn.Sigmoid()
            )

        # --- 线性映射层 (无偏置) ---
        # 设置 bias=False，因为 CAC 要求特征以原点为参考，严格对齐到坐标轴上的锚点
        self.classify = nn.Linear(input_dim, num_classes, bias=False)

        # --- 初始化锚点 (Anchors) ---
        if anchor_mode == 'axis_aligned':
            # 原始论文实现: 锚点在正坐标轴上 [num_classes, num_classes]
            anchors = torch.eye(num_classes) * alpha

        elif anchor_mode == 'uniform_hypersphere':
            # 锚点均匀分布在整个超球面空间
            # 1. 生成随机正态分布向量 (高维空间中高斯分布近似均匀分布)
            rand_anchors = torch.randn(num_classes, num_classes)
            # 2. 正交化: 确保锚点之间尽可能垂直，最大化区分度。
            q, r = torch.linalg.qr(rand_anchors)
            anchors = q.t()  # 使行向量是锚点

            # 3. 归一化并缩放到 alpha
            anchors = F.normalize(anchors, p=2, dim=1) * alpha
        elif anchor_mode == 'negative_shattered':
            anchors = torch.full((num_classes, num_classes), -self.alpha * 0.1)
            anchors.fill_diagonal_(self.alpha)
        else:
            raise ValueError("Unknown anchor_mode")
        self.register_buffer('anchors', anchors) # 随模型保存/加载，但不会被优化器更新


    def forward(self, x):
        if self.se_reduction > 0:
            # 计算每个特征维度的重要性权重 (0~1)
            attention_weights = self.se_block(x)
            # 加权原始特征 (Scale)
            x = x * attention_weights

        logits = self.classify(x)

        # 计算距离矩阵 (Euclidean Distance)
        logits_expand = logits.unsqueeze(1) # [B, N] -> [B, 1, N]
        anchors_expand = self.anchors.unsqueeze(0)  # [N, N] -> [1, N, N]

        # 计算欧几里得距离 ||z - c||_2 (每个样本到每个类中心的距离)
        distances = torch.norm(logits_expand - anchors_expand, p=2, dim=2)  # [B, N]

        return logits, distances


class CACLoss(nn.Module):
    """
    CAC 损失函数: L_CAC = L_T + lambda * L_A
    """
    def __init__(self, lambda_w=0.1):
        super(CACLoss, self).__init__()
        self.lambda_w = lambda_w

    def forward(self, distances, targets):
        """
        distances: [Batch, Num_Classes]
        targets: [Batch] (类别索引)
        """
        # --- Anchor Loss (L_A) ---
        # 目标：最小化样本与其真实类别锚点之间的距离
        true_class_distances = distances.gather(1, targets.unsqueeze(1)).squeeze(1)  # 选 target 对应的那个距离值
        l_anchor = torch.mean(true_class_distances)

        # --- Tuplet Loss (L_T) ---
        # 目标：最大化样本与其他错误锚点的距离
        # 论文技巧：对负距离求 CrossEntropy 等价于 Softmin 概率逻辑
        l_tuplet = F.cross_entropy(-distances, targets)

        # --- 总损失 ---
        return l_tuplet + self.lambda_w * l_anchor


if __name__ == "__main__":
    import numpy as np
    model = CACProjector(input_dim=100, num_classes=10, alpha=10.0, anchor_mode='uniform_hypersphere')
    print(np.array(model.anchors.tolist()))
    print(model.anchors.shape)