import torch.nn as nn


class LinearClassifier(nn.Module):
    """线性分类器模型"""
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


class HierarchicalClassifier(nn.Module):
    """层次分类器模型，支持 Soft Attention"""
    def __init__(self, feature_dim, num_super, num_sub, use_attention=True):
        super(HierarchicalClassifier, self).__init__()
        self.super_head = nn.Linear(feature_dim, num_super)
        self.sub_head = nn.Linear(feature_dim, num_sub)
        self.use_attention = use_attention
        
        if use_attention:
            # SE-style attention: super_logits -> feature weights
            self.attention = nn.Sequential(
                nn.Linear(num_super, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
    
    def forward(self, features):
        super_logits = self.super_head(features)
        
        if self.use_attention:
            attn_weights = self.attention(super_logits)
            features = features * attn_weights
        
        sub_logits = self.sub_head(features)
        return super_logits, sub_logits
