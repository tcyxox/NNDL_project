import torch.nn as nn


class LinearClassifier(nn.Module):
    """线性分类器模型"""
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)
