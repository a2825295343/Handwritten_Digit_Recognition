import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """全连接神经网络（多层感知机）"""
    def __init__(self):
        super(MLP, self).__init__()
        # 网络结构：输入层(784) → 隐藏层1(512) → 隐藏层2(256) → 输出层(10)
        self.fc1 = nn.Linear(28*28, 512)  # 第一层全连接
        self.fc2 = nn.Linear(512, 256)    # 第二层全连接
        self.fc3 = nn.Linear(256, 10)     # 输出层（10个数字类别）
        self.dropout = nn.Dropout(0.2)    # 防止过拟合

    def forward(self, x):
        # 展平输入：(batch, 1, 28, 28) → (batch, 784)
        x = x.view(-1, 28*28)
        # 前向传播（激活函数+dropout）
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_params_count(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)