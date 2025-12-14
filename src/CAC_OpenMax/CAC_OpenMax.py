import numpy as np
import torch
import torch.nn as nn
import libmr
import scipy.spatial.distance as spd


class CAC_OpenMax:
    def __init__(self, num_classes, cac_projector=None, weibul_tail_size=3, alpha=3, distance_type='euclidean'):
        """
        初始化 OpenMax
        :param num_classes: 已知类别的数量 (N)
        :param weibul_tail_size: 用于拟合 Weibull 分布的尾部大小
        :param alpha: 修正 Top-K 个类别的得分
        :param distance_type: 距离度量方式 ('euclidean', 'cosine', 'euclidean_cosine')
        """
        self.num_classes = num_classes
        self.tail_size = weibul_tail_size
        self.alpha = alpha
        self.distance_type = distance_type

        # 存储每个类别的 anchor 作为 MAV
        # self.mavs = cac_projector.anchors.cpu().numpy()
        self.mavs = None
        # 存储每个类别的 Weibull 模型
        self.weibull_models = {}

    def fit(self, activation_vectors, labels):
        """
        校准: 利用分类正确的训练数据计算 MAV 和 Weibull 模型
        """
        # 将输入转为 numpy 格式
        if isinstance(activation_vectors, torch.Tensor):
            activation_vectors = activation_vectors.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        self.mavs = np.zeros((self.num_classes, activation_vectors.shape[1]))

        # print("开始 OpenMax 校准 (Fit)...")
        for c in range(self.num_classes):
            # 1. 获取该类别下预测**正确**的样本（先过滤出 predict == label 的数据）
            class_idxs = (labels == c)
            class_acts = activation_vectors[class_idxs]

            # 2. 计算 Mean Activation Vector (MAV)
            self.mavs[c] = np.mean(class_acts, axis=0)

            # 3. 计算所有样本到 MAV 的距离
            dists = self._compute_distance(class_acts, self.mavs[c])

            # 4. 拟合 Weibull 分布 (只取最大的 tail_size 个距离)
            # 极值理论关注的是“最远”的那些点
            mr = libmr.MR()
            tail_dists = sorted(dists)[-self.tail_size:]
            mr.fit_high(tail_dists, len(tail_dists))
            self.weibull_models[c] = mr

        # print("校准完成。")

    def predict(self, activation_vectors):
        """
        推理阶段: 计算包含“未知类”的概率分布
        """
        if isinstance(activation_vectors, torch.Tensor):
            activation_vectors = activation_vectors.cpu().detach().numpy()

        N = activation_vectors.shape[0]
        # 输出维度是 N x (num_classes + 1)，因为多了第 0 类 (Unknown)
        openmax_probs = np.zeros((N, self.num_classes + 1))

        for i in range(N):
            act = activation_vectors[i]

            # 1. 找到激活值最大的 Top-alpha 个类别
            # argsort 返回从小到大，取最后 alpha 个并反转
            sorted_indices = np.argsort(act)[::-1]
            alpha_indices = sorted_indices[:self.alpha]

            # 权重向量，初始为 1
            weights = np.ones(self.num_classes)

            # 2. 对 Top-alpha 类别进行修正
            for rank, class_idx in enumerate(alpha_indices):
                # 获取对应的 MAV 和 Weibull 模型
                mav = self.mavs[class_idx]
                weibull_model = self.weibull_models[class_idx]

                # 计算当前输入与该类 MAV 的距离
                dist = self._compute_distance(act[None, :], mav)[0]

                # 计算属于“异常值”的概率 (weibull_cdf)
                w_score = weibull_model.w_score(dist)

                # 计算修正权重: 结合了 rank 和异常概率
                # 论文公式: ((alpha - rank) / alpha) * e^... 或者是直接用 w_score
                # 这里采用标准的 OpenMax 权重计算方式:
                modified_score = act[class_idx] * (1 - w_score * ((self.alpha - rank) / self.alpha))

                # 更新权重 (用于后续计算未知类概率)
                # 实际上 OpenMax 论文中的做法是直接修改激活值
                weights[class_idx] = 1 - w_score * ((self.alpha - rank) / self.alpha)

            # 3. 计算修正后的激活向量 (Revised Activation Vector)
            revised_act = act * weights

            # 4. 计算“未知类”的伪激活值 (Pseudo-activation)
            # 未知类的激活值 = 所有已知类被削减掉的激活值之和
            unknown_act = np.sum(act * (1 - weights))

            # 5. 组合最终向量 [Unknown, Class_1, Class_2, ...]
            # 论文中建议 Unknown 放在索引 0
            final_act = np.insert(revised_act, 0, unknown_act)

            # 6. 计算 Softmax
            openmax_probs[i] = self._softmax(final_act)

        return openmax_probs

    def _compute_distance(self, X, center):
        """计算距离辅助函数"""
        # X: (N, features), center: (features,)
        if self.distance_type == 'euclidean':
            return spd.cdist(X, center[None, :], metric='euclidean').ravel()
        elif self.distance_type == 'cosine':
            return spd.cdist(X, center[None, :], metric='cosine').ravel()
        elif self.distance_type == 'euclidean_cosine':
            # 论文补充材料中提到的组合距离
            d_euc = spd.cdist(X, center[None, :], metric='euclidean').ravel()
            d_cos = spd.cdist(X, center[None, :], metric='cosine').ravel()
            return d_euc + d_cos  # 简单的线性组合
        else:
            raise ValueError("Unknown distance type")

    # TODO
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class CAC_OpenMax_System(nn.Module):
    def __init__(self, CAC_model, openmax_model, device):
        super().__init__()
        self.CAC_model = CAC_model
        self.openmax_model = openmax_model
        self.device = device

    def predict(self, features):
        """
        输入特征，输出预测结果
        返回: (预测的类别索引, OpenMax概率矩阵)
        注意:
        - 类别 0 代表 "Unknown"
        - 类别 1 代表 "Known Class 0", 类别 2 代表 "Known Class 1"...
        """
        self.CAC_model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            # 1. 获取激活向量 (Logits)
            logits, _ = self.CAC_model(features)

            # 2. OpenMax 推理
            # predict 返回的是 N x (num_classes + 1) 的概率
            probs = self.openmax_model.predict(logits)

            # 3. 获取预测类别 (0是未知, >0是已知)
            preds = np.argmax(probs, axis=1)

        return preds, probs