import torch
import torch.nn as nn
import torch.nn.functional as F

class MILHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512):
        super(MILHead, self).__init__()
        
        # 特征转换层
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, num_instances, features)
        batch_size, num_instances, _ = x.size()
        
        # 特征转换
        h = self.feature_extractor(x)  # (batch_size, num_instances, hidden_dim)
        
        # 计算注意力权重
        a = self.attention(h)  # (batch_size, num_instances, 1)
        a = torch.softmax(a, dim=1)  # 在实例维度上进行softmax
        
        # 加权聚合
        m = torch.sum(a * h, dim=1)  # (batch_size, hidden_dim)
        
        # 分类
        logits = self.classifier(m)  # (batch_size, num_classes)
        
        return logits, a  # 返回分类结果和注意力权重 