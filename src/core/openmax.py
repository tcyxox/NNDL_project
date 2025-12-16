"""
OpenMax 模块 - 使用 Weibull 分布进行 Open Set Recognition

基于 scipy.stats.weibull_min 实现，无需额外依赖

References:
    Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. CVPR.
"""
import numpy as np
import scipy.spatial.distance as spd
from scipy.stats import weibull_min
import torch


class OpenMax:
    """
    OpenMax 开放集识别模块
    
    通过拟合 Weibull 分布来检测未知类样本
    """
    
    def __init__(self, num_classes, weibull_tail_size=20, alpha=5, distance_type='euclidean'):
        """
        初始化 OpenMax
        
        Args:
            num_classes: 已知类别的数量 (N)
            weibull_tail_size: 用于拟合 Weibull 分布的尾部大小
            alpha: 修正 Top-K 个类别的得分
            distance_type: 距离度量方式 ('euclidean', 'cosine', 'euclidean_cosine')
        """
        self.num_classes = num_classes
        self.tail_size = weibull_tail_size
        self.alpha = alpha
        self.distance_type = distance_type

        # 存储每个类别的 Mean Activation Vector (MAV)
        self.mavs = None
        # 存储每个类别的 Weibull 模型参数 (shape, loc, scale)
        self.weibull_params = {}

    def fit(self, activation_vectors, labels):
        """
        校准: 利用分类正确的训练数据计算 MAV 和 Weibull 模型
        
        Args:
            activation_vectors: [N, C] - 模型输出的 logits
            labels: [N] - 真实标签 (local id, 0 ~ num_classes-1)
        """
        # 将输入转为 numpy 格式
        if isinstance(activation_vectors, torch.Tensor):
            activation_vectors = activation_vectors.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        self.mavs = np.zeros((self.num_classes, activation_vectors.shape[1]))

        for c in range(self.num_classes):
            # 1. 获取该类别的样本
            class_idxs = (labels == c)
            class_acts = activation_vectors[class_idxs]
            
            if len(class_acts) == 0:
                print(f"警告: 类别 {c} 没有样本，跳过")
                continue

            # 2. 计算 Mean Activation Vector (MAV)
            self.mavs[c] = np.mean(class_acts, axis=0)

            # 3. 计算所有样本到 MAV 的距离
            dists = self._compute_distance(class_acts, self.mavs[c])

            # 4. 拟合 Weibull 分布 (只取最大的 tail_size 个距离)
            # 极值理论关注的是"最远"的那些点
            tail_size = min(self.tail_size, len(dists))
            tail_dists = np.sort(dists)[-tail_size:]
            
            # 使用 scipy 拟合 Weibull 分布
            try:
                shape, loc, scale = weibull_min.fit(tail_dists, floc=0)
                self.weibull_params[c] = (shape, loc, scale)
            except Exception as e:
                print(f"警告: 类别 {c} Weibull 拟合失败: {e}")
                # 使用默认参数
                self.weibull_params[c] = (1.0, 0.0, np.mean(tail_dists) if len(tail_dists) > 0 else 1.0)

    def predict(self, activation_vectors):
        """
        推理阶段: 计算包含"未知类"的概率分布
        
        Args:
            activation_vectors: [N, C] - 模型输出的 logits
            
        Returns:
            openmax_probs: [N, num_classes + 1] - 概率分布，第 0 列是 unknown 概率
        """
        if isinstance(activation_vectors, torch.Tensor):
            activation_vectors = activation_vectors.cpu().detach().numpy()

        N = activation_vectors.shape[0]
        # 输出维度是 N x (num_classes + 1)，因为多了第 0 类 (Unknown)
        openmax_probs = np.zeros((N, self.num_classes + 1))

        for i in range(N):
            act = activation_vectors[i]

            # 1. 找到激活值最大的 Top-alpha 个类别
            sorted_indices = np.argsort(act)[::-1]
            alpha_indices = sorted_indices[:self.alpha]

            # 权重向量，初始为 1
            weights = np.ones(self.num_classes)

            # 2. 对 Top-alpha 类别进行修正
            for rank, class_idx in enumerate(alpha_indices):
                if class_idx >= self.num_classes or class_idx not in self.weibull_params:
                    continue
                    
                # 获取对应的 MAV 和 Weibull 模型参数
                mav = self.mavs[class_idx]
                shape, loc, scale = self.weibull_params[class_idx]

                # 计算当前输入与该类 MAV 的距离
                dist = self._compute_distance(act[None, :], mav)[0]

                # 计算属于"异常值"的概率 (Weibull CDF)
                w_score = weibull_min.cdf(dist, shape, loc=loc, scale=scale)

                # 计算修正权重
                weights[class_idx] = 1 - w_score * ((self.alpha - rank) / self.alpha)

            # 3. 计算修正后的激活向量 (Revised Activation Vector)
            revised_act = act * weights

            # 4. 计算"未知类"的伪激活值 (Pseudo-activation)
            # 未知类的激活值 = 所有已知类被削减掉的激活值之和
            unknown_act = np.sum(act * (1 - weights))

            # 5. 组合最终向量 [Unknown, Class_1, Class_2, ...]
            final_act = np.insert(revised_act, 0, unknown_act)

            # 6. 计算 Softmax
            openmax_probs[i] = self._softmax(final_act)

        return openmax_probs

    def get_unknown_probs(self, activation_vectors):
        """
        获取 unknown 概率 (第 0 列)
        
        Args:
            activation_vectors: [N, C] - 模型输出的 logits
            
        Returns:
            unknown_probs: [N] - unknown 概率
        """
        probs = self.predict(activation_vectors)
        return probs[:, 0]

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
            return d_euc + d_cos
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def fit_openmax_for_model(model, features, labels, label_map, openmax_config, device):
    """
    为模型拟合 OpenMax
    
    Args:
        model: 分类器模型 (LinearSingleHead 或 GatedDualHead 的一个 head)
        features: 训练特征
        labels: 训练标签 (global id)
        label_map: global_to_local 映射
        openmax_config: OpenMax 配置
        device: 设备
        
    Returns:
        openmax: 拟合好的 OpenMax 对象
    """
    model.eval()
    
    # 获取 logits
    with torch.no_grad():
        features_dev = features.to(device)
        logits = model(features_dev)
        if isinstance(logits, tuple):
            # GatedDualHead 返回 (super_logits, sub_logits)
            logits = logits[0]  # 默认取第一个
    
    # 转换标签为 local id
    labels_local = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)
    
    # 创建并拟合 OpenMax
    num_classes = len(label_map)
    openmax = OpenMax(
        num_classes=num_classes,
        weibull_tail_size=openmax_config.weibull_tail_size,
        alpha=openmax_config.alpha,
        distance_type=openmax_config.distance_type
    )
    
    openmax.fit(logits, labels_local)
    
    return openmax
