import torch.nn as nn


class LinearSingleHead(nn.Module):
    """线性分类器模型"""
    def __init__(self, in_features, out_features):
        super(LinearSingleHead, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


class GatedDualHead(nn.Module):
    """层次分类器模型，使用 SE Attention"""
    def __init__(self, feature_dim, num_super, num_sub):
        super(GatedDualHead, self).__init__()
        self.super_head = nn.Linear(feature_dim, num_super)
        self.sub_head = nn.Linear(feature_dim, num_sub)
        
        # SE-style attention: feature -> squeeze -> excite -> feature weights
        reduction = 4
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),  # Squeeze
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),  # Excitation
            nn.Sigmoid()
        )
    
    def forward(self, features):
        super_logits = self.super_head(features)
        
        # SE Attention
        attn_weights = self.attention(features)
        attended_features = features * attn_weights
        sub_logits = self.sub_head(attended_features)
        
        return super_logits, sub_logits
