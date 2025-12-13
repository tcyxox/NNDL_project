import json
import os

import torch
import torch.nn.functional as F

from .models import LinearSingleHead, GatedDualHead


def compute_energy(logits, temperature):
    """
    计算 Energy Score: E(x; T) = -T × log(Σ exp(logit_i / T))

    Args:
        logits: [N, C] - 模型输出的 logits
        temperature: float - 温度缩放参数
            - T = 0: 使用 -max(logits) (极限情况)
            - T < 1: 锐化（扩大差距）
            - T = 1: 不变
            - T > 1: 软化（缩小差距）

    Returns:
        energy: [N] - 能量得分，越低越可能是已知类
    """
    if temperature == 0:
        # 当 T → 0 时，lim_{T→0} -T × log(Σ exp(logit_i / T)) = -max(logits)
        return -torch.max(logits, dim=1)[0]
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def calculate_threshold_linear_single_head(
        model, val_features, val_labels, label_map,
        target_recall, device,
        temperature, use_energy, use_sigmoid_bce
):
    """
    在验证集上为 LinearSingleHead 计算 OSR 阈值

    Args:
        model: LinearSingleHead 模型
        val_features: 验证集特征
        val_labels: 验证集标签
        label_map: local_to_global 映射
        target_recall: 目标召回率 (如 0.95)
        device: 'cuda' or 'cpu'
        temperature: OOD 温度缩放参数
        use_energy: 是否使用 energy score（否则用 MSP）
        use_sigmoid_bce: 是否使用 sigmoid-based scoring

    Returns:
        threshold: 计算出的阈值
    """
    model.eval()

    # 筛选出验证集里的"已知类"样本
    known_mask = torch.tensor([l.item() in label_map.values() for l in val_labels])
    X_known = val_features[known_mask].to(device)

    if len(X_known) == 0:
        print("警告: 验证集中没有已知类样本，使用默认阈值")
        return 0.5 if use_sigmoid_bce else (-5 if use_energy else 0.5)

    with torch.no_grad():
        logits = model(X_known)
        if use_energy:
            energy = compute_energy(logits, temperature)  # 越低越可能是已知类，阈值是能量上限
            threshold = torch.quantile(energy, target_recall).item()
        else:
            if use_sigmoid_bce:
                # Maximum Sigmoid
                scores = torch.max(torch.sigmoid(logits / temperature), dim=1)[0]
                threshold = torch.quantile(scores, 1 - target_recall).item()
            else:
                # MSP with temperature = ODIN
                probs = F.softmax(logits / temperature, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                threshold = torch.quantile(max_probs, 1 - target_recall).item()

    return threshold


def calculate_threshold_gated_dual_head(
        model, val_features, val_super_labels, val_sub_labels,
        super_map_inv, sub_map_inv, target_recall, device,
        temperature, use_energy, use_sigmoid_bce
):
    """
    在验证集上为 GatedDualHead 计算 OSR 阈值

    Args:
        model: GatedDualHead 模型
        val_features: [N, D] - 验证集特征
        val_super_labels: [N] - 验证集超类标签
        val_sub_labels: [N] - 验证集子类标签
        super_map_inv: {local_id: global_id} - 超类映射
        sub_map_inv: {local_id: global_id} - 子类映射
        target_recall: float - 目标召回率 (如 0.95)
        device: str - 'cuda' or 'cpu'
        temperature: float - OOD 温度缩放参数
        use_energy: bool - 是否使用 energy score（否则用 MSP）
        use_sigmoid_bce: bool - 是否使用 sigmoid-based scoring

    Returns:
        thresh_super: float - 超类阈值
        thresh_sub: float - 子类阈值
    """

    def _calculate_single_threshold(logits, known_mask, target_recall, temperature, use_energy, use_sigmoid_bce):
        """
        Args:
            logits: [N, C] - 模型输出的 logits
            known_mask: [N] - 布尔掩码，标记已知类样本
            target_recall: float - 目标召回率
            temperature: float - 温度缩放参数
            use_energy: bool - 是否使用 energy score
            use_sigmoid_bce: bool - 是否使用 sigmoid-based scoring

        Returns:
            threshold: float - 计算出的阈值
        """
        if known_mask.sum() > 0:
            if use_energy:
                scores = compute_energy(logits[known_mask], temperature)  # [N_known, C] -> [N_known]
                threshold = torch.quantile(scores, target_recall).item()  # [N_known]
            else:
                if use_sigmoid_bce:
                    # Maximum Sigmoid
                    scores = torch.max(torch.sigmoid(logits[known_mask] / temperature), dim=1)[
                        0]  # [N_known, C] -> [N_known]
                    threshold = torch.quantile(scores, 1 - target_recall).item()  # [N_known]
                else:
                    # MSP with temperature = ODIN
                    probs = F.softmax(logits[known_mask] / temperature, dim=1)  # [N_known, C] -> [N_known, C]
                    threshold = torch.quantile(probs.max(dim=1)[0], 1 - target_recall).item()  # [N_known]
        else:
            print(f"Warning: No known samples in validation set, using default threshold")
            threshold = 0.5 if use_sigmoid_bce else (-5 if use_energy else 0.5)

        return threshold

    model.eval()

    with torch.no_grad():
        super_logits, sub_logits = model(val_features.to(device))  # [N, D] -> [N, 3], [N, 70]

    # 计算 superclass threshold
    known_super = torch.tensor([l.item() in super_map_inv for l in val_super_labels])  # [N]
    thresh_super = _calculate_single_threshold(super_logits, known_super, target_recall, temperature, use_energy,
                                               use_sigmoid_bce)

    # 计算 subclass threshold
    known_sub = torch.tensor([l.item() in sub_map_inv for l in val_sub_labels])  # [N]
    thresh_sub = _calculate_single_threshold(sub_logits, known_sub, target_recall, temperature, use_energy,
                                             use_sigmoid_bce)

    return thresh_super, thresh_sub


def load_linear_single_head(prefix, model_dir, feature_dim, device):
    """
    Args:
        prefix: 'super' or 'sub'
        model_dir: 模型目录
        feature_dim: 特征维度
        device: 'cuda' or 'cpu'

    Returns:
        model: 加载好的模型
        local_to_global: 映射字典 (模型内部ID -> 原始ID)
    """
    # 1. 加载映射表 (Local ID -> Global ID)
    mapping_path = os.path.join(model_dir, f"{prefix}_local_to_global_map.json")
    with open(mapping_path, 'r') as f:
        local_to_global = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(local_to_global)
    print(f"[{prefix}] 加载映射表: 检测到 {num_classes} 个已知类")

    # 2. 初始化模型
    model = LinearSingleHead(feature_dim, num_classes)
    model_path = os.path.join(model_dir, f"{prefix}_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, local_to_global


def load_gated_dual_head(model_dir, feature_dim, num_super, num_sub, device):
    """
    Args:
        model_dir: 模型目录
        feature_dim: 特征维度
        num_super: 超类数量
        num_sub: 子类数量
        device: 'cuda' or 'cpu'

    Returns:
        model: 加载好的模型
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
    """

    # 加载映射表
    with open(os.path.join(model_dir, "super_local_to_global_map.json"), 'r') as f:
        super_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(model_dir, "sub_local_to_global_map.json"), 'r') as f:
        sub_map = {int(k): v for k, v in json.load(f).items()}

    print(f"[hierarchical] 加载映射表: {len(super_map)} 超类, {len(sub_map)} 子类")

    # 初始化并加载模型
    model = GatedDualHead(feature_dim, num_super, num_sub)
    model_path = os.path.join(model_dir, "hierarchical_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model, super_map, sub_map


def predict_with_linear_single_head(
        features, super_model, sub_model,
        super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx, device,
        super_to_sub, temperature, use_energy, use_sigmoid_bce
):
    """
    Args:
        features: 输入特征 [N, 512]
        super_model: 超类分类模型
        sub_model: 子类分类模型
        super_map: 超类 local_to_global 映射
        sub_map: 子类 local_to_global 映射
        thresh_super: 超类阈值
        thresh_sub: 子类阈值
        novel_super_idx: 未知超类的 ID (3)
        novel_sub_idx: 未知子类的 ID (87)
        device: 'cuda' or 'cpu'
        super_to_sub: 超类到子类的映射 {super_id: [sub_ids]}（可选，用于 masking）
        temperature: OOD 温度缩放参数
        use_energy: 是否使用 energy score（否则用 MSP）
        use_sigmoid_bce: 是否使用 sigmoid-based scoring

    Returns:
        super_preds: 超类预测列表 [N]
        sub_preds: 子类预测列表 [N]
        super_scores: 超类 OOD 得分列表 [N] (用于 AUROC，越高越像已知类)
        sub_scores: 子类 OOD 得分列表 [N] (用于 AUROC，越高越像已知类)
    """
    super_preds = []
    sub_preds = []
    super_scores = []  # OOD 得分（用于 AUROC）
    sub_scores = []  # OOD 得分（用于 AUROC）

    # 是否启用 hierarchical masking
    use_masking = (super_to_sub is not None)
    num_sub_classes = sub_model.layer.out_features

    # 如果启用 masking，从 sub_map 反转计算 global_to_local
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)

            # === 超类预测 ===
            super_logits = super_model(feature)
            if use_sigmoid_bce:
                super_probs = torch.sigmoid(super_logits)
            else:
                super_probs = F.softmax(super_logits / temperature, dim=1)
            max_super_prob, super_idx = torch.max(super_probs, dim=1)

            # 计算 OOD 得分 (Energy 优先)
            if use_energy:
                energy = compute_energy(super_logits, temperature).item()
                super_score = -energy  # 取负，Energy 越低越像已知类
            else:
                super_score = max_super_prob.item()  # max sigmoid 或 MSP

            super_scores.append(super_score)

            # 判断是否为 novel
            if use_energy:
                is_novel_super = energy > thresh_super
            else:
                is_novel_super = max_super_prob.item() < thresh_super

            if is_novel_super:
                final_super = novel_super_idx
            else:
                final_super = super_map[super_idx.item()]

            # === 子类预测（带 Hierarchical Masking）===
            sub_logits = sub_model(feature)

            # 计算 OOD 得分 （必须在 masking 之前，与阈值计算保持一致）
            # Energy 优先
            if use_energy:
                energy = compute_energy(sub_logits, temperature).item()
                sub_score = -energy  # 取负，Energy 越低越像已知类
            elif use_sigmoid_bce:
                sub_probs_unmasked = torch.sigmoid(sub_logits)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()
            else:
                # MSP with temperature = ODIN
                sub_probs_unmasked = F.softmax(sub_logits / temperature, dim=1)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()

            sub_scores.append(sub_score)

            # 如果超类不是 novel 且启用了 masking，则 mask 掉不属于该超类的子类
            if use_masking and final_super != novel_super_idx and final_super in super_to_sub:
                valid_subs = super_to_sub[final_super]
                mask = torch.full((1, num_sub_classes), float('-inf'), device=device)
                for sub_id in valid_subs:
                    if sub_id in sub_global_to_local:
                        local_id = sub_global_to_local[sub_id]
                        mask[0, local_id] = 0
                sub_logits = sub_logits + mask

            # 对 masked logits 计算最终预测
            if use_sigmoid_bce:
                sub_probs = torch.sigmoid(sub_logits)
            else:
                sub_probs = F.softmax(sub_logits, dim=1)
            _, sub_idx = torch.max(sub_probs, dim=1)

            # 判断是否为 novel（使用 masking 前计算的分数）
            if use_energy:
                is_novel_sub = energy > thresh_sub
            else:
                is_novel_sub = sub_score < thresh_sub

            if is_novel_sub:
                final_sub = novel_sub_idx
            else:
                final_sub = sub_map[sub_idx.item()]

            # === Hard Constraint: 超类 novel → 子类也 novel ===
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx

            super_preds.append(final_super)
            sub_preds.append(final_sub)

    return super_preds, sub_preds, super_scores, sub_scores


def predict_with_gated_dual_head(
        features, model, super_map, sub_map,
        thresh_super, thresh_sub,
        novel_super_idx, novel_sub_idx, device,
        super_to_sub, temperature, use_energy, use_sigmoid_bce
):
    """
    Args:
        features: [N, D] - 输入特征
        model: GatedDualHead 模型
        super_map: {local_id: global_id} - 超类映射
        sub_map: {local_id: global_id} - 子类映射
        thresh_super: float - 超类阈值
        thresh_sub: float - 子类阈值
        novel_super_idx: int - 未知超类的 ID (3)
        novel_sub_idx: int - 未知子类的 ID (87)
        device: str - 'cuda' or 'cpu'
        super_to_sub: dict - 超类到子类的映射（用于 hard masking，可选）
        temperature: float - OOD 温度缩放参数
        use_energy: bool - 是否使用 energy score（否则用 MSP）
        use_sigmoid_bce: bool - 是否使用 sigmoid-based scoring

    Returns:
        super_preds: list[int] - 超类预测列表 [N]
        sub_preds: list[int] - 子类预测列表 [N]
        super_scores: list[float] - 超类 OOD 得分 [N] (越高越像已知类)
        sub_scores: list[float] - 子类 OOD 得分 [N] (越高越像已知类)
    """
    super_preds = []
    sub_preds = []
    super_scores = []  # OOD 得分（用于 AUROC）
    sub_scores = []  # OOD 得分（用于 AUROC）

    use_masking = (super_to_sub is not None)
    num_sub_classes = len(sub_map)
    sub_global_to_local = {v: int(k) for k, v in sub_map.items()} if use_masking else None

    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)

            # 模型同时输出 super 和 sub logits
            super_logits, sub_logits = model(feature)

            # === 超类预测 ===
            # Step 1: 计算概率分布
            if use_sigmoid_bce:
                super_probs = torch.sigmoid(super_logits)
            else:
                super_probs = F.softmax(super_logits / temperature, dim=1)
            max_super_prob, super_idx = torch.max(super_probs, dim=1)

            # Step 2: 计算 OOD 得分
            if use_sigmoid_bce:
                super_score = max_super_prob.item()
            elif use_energy:
                energy = compute_energy(super_logits, temperature).item()
                super_score = -energy  # 取负，Energy 越低越像已知类
            else:
                super_score = max_super_prob.item()  # MSP 越高越像已知类

            super_scores.append(super_score)

            # Step 3: 判断是否为 novel
            if use_sigmoid_bce:
                is_novel_super = max_super_prob.item() < thresh_super
            elif use_energy:
                is_novel_super = energy > thresh_super
            else:
                is_novel_super = max_super_prob.item() < thresh_super

            # Step 4: 确定最终预测
            if is_novel_super:
                final_super = novel_super_idx
            else:
                final_super = super_map[super_idx.item()]

            # === 子类预测（带 Hierarchical Masking）===

            # Step 1: 计算 OOD 得分（必须在 masking 之前，与阈值计算保持一致）
            if use_sigmoid_bce:
                sub_probs_unmasked = torch.sigmoid(sub_logits)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()
            elif use_energy:
                energy = compute_energy(sub_logits, temperature).item()
                sub_score = -energy  # 取负，Energy 越低越像已知类
            else:
                sub_probs_unmasked = F.softmax(sub_logits / temperature, dim=1)
                sub_score = sub_probs_unmasked.max(dim=1)[0].item()  # MSP 越高越像已知类

            sub_scores.append(sub_score)

            # Step 2: 判断是否为 novel（使用 masking 前计算的分数）
            if use_sigmoid_bce:
                is_novel_sub = sub_score < thresh_sub
            elif use_energy:
                is_novel_sub = energy > thresh_sub
            else:
                is_novel_sub = sub_score < thresh_sub

            # Step 3: 如果超类不是 novel 且启用了 masking，则 mask 掉不属于该超类的子类
            if use_masking and final_super != novel_super_idx and final_super in super_to_sub:
                valid_subs = super_to_sub[final_super]
                mask = torch.full((1, num_sub_classes), float('-inf'), device=device)
                for sub_id in valid_subs:
                    if sub_id in sub_global_to_local:
                        local_id = sub_global_to_local[sub_id]
                        mask[0, local_id] = 0
                sub_logits = sub_logits + mask

            # Step 4: 对 masked logits 计算最终预测
            if use_sigmoid_bce:
                sub_probs = torch.sigmoid(sub_logits)
            else:
                sub_probs = F.softmax(sub_logits, dim=1)
            _, sub_idx = torch.max(sub_probs, dim=1)

            if is_novel_sub:
                final_sub = novel_sub_idx
            else:
                final_sub = sub_map[sub_idx.item()]

            # === Hard Constraint: 超类 novel → 子类也 novel ===
            if final_super == novel_super_idx:
                final_sub = novel_sub_idx

            super_preds.append(final_super)
            sub_preds.append(final_sub)

    return super_preds, sub_preds, super_scores, sub_scores
