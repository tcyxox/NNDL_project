import torch
import torch.nn as nn
import torch.nn.functional as F


class CACProjector(nn.Module):
    """
    CAC (Class Anchor Clustering) 投影头
    用于将 CLIP 特征映射到符合 CAC 几何约束的 Logit 空间。
    """
    def __init__(self, input_dim, num_classes, alpha=10.0, se_reduction=-1, anchor_mode="axis_aligned"):
        super(CACProjector, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.se_reduction = se_reduction

        # --- SE Attention Module ---
        if self.se_reduction > 0:
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
            # 锚点在正坐标轴上 [num_classes, num_classes]
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
        CAC Loss 升级版: 支持 Open Set 样本 (OpenMix)
    """
    def __init__(self, lambda_w=0.1, lambda_open=0.1):
        """
        lambda_w: Anchor loss 的权重
        lambda_open: 未知样本损失的权重
        """
        super(CACLoss, self).__init__()
        self.lambda_w = lambda_w
        self.lambda_open = lambda_open

    def forward(self, logits, distances, targets):
        """
        logits: [Batch, Num_Classes] -> 需要用到 logits本身来计算到原点的距离
        distances: [Batch, Num_Classes] -> CACProjector 输出的距离矩阵
        targets: [Batch] -> 标签
        """

        # 1. 区分已知类和未知类样本
        # 创建掩码
        known_mask = targets != -1
        unknown_mask = ~known_mask

        # ================== 处理已知类 (Known) ==================
        loss_known = torch.tensor(0.0, device=logits.device)
        if known_mask.sum() > 0:
            known_dist = distances[known_mask]
            known_targets = targets[known_mask]

            # L_Anchor: 拉近到正确锚点
            true_class_distances = known_dist.gather(1, known_targets.unsqueeze(1)).squeeze(1)
            l_anchor = torch.mean(true_class_distances)

            # L_Tuplet: 远离错误锚点 (Softmax/CrossEntropy)
            l_tuplet = F.cross_entropy(-known_dist, known_targets)

            loss_known = l_tuplet + self.lambda_w * l_anchor

        # ================== 处理未知类 (Open/Generated) ==================
        # 策略：最小化到原点的距离 (即最小化 logits 的模长)
        loss_open = torch.tensor(0.0, device=logits.device)
        if unknown_mask.sum() > 0:
            unknown_logits = logits[unknown_mask]

            # 计算 Logits 的 L2 范数 (即到原点的距离)
            # 目标是让 output 趋向于 0 向量
            loss_open = torch.mean(torch.norm(unknown_logits, p=2, dim=1))
            # 或者是用平方和 (MSE to 0):
            # loss_open = torch.mean(unknown_logits ** 2)

        # ================== 总损失 ==================
        return loss_known + self.lambda_open * loss_open


if __name__ == "__main__":
    import numpy as np
    model = CACProjector(input_dim=100, num_classes=10, alpha=10.0, anchor_mode='uniform_hypersphere')
    print(np.array(model.anchors.tolist()))
    print(model.anchors.shape)