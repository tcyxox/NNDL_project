import torch
import torch.nn as nn
import torch.nn.functional as F


class CACProjector(nn.Module):
    """
    CAC (Class Anchor Clustering) 投影头
    用于将 CLIP 特征映射到符合 CAC 几何约束的 Logit 空间。
    """
    def __init__(self, input_dim, num_classes, alpha=10.0):
        super(CACProjector, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        # 线性映射层 (无偏置)
        # 设置 bias=False，因为 CAC 要求特征以原点为参考，严格对齐到坐标轴上的锚点
        self.classify = nn.Linear(input_dim, num_classes, bias=False)

        # 初始化锚点 (Anchors)
        anchors = torch.eye(num_classes) * alpha
        self.register_buffer('anchors', anchors)    # 随模型保存/加载，但不会被优化器更新

    def forward(self, x):
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