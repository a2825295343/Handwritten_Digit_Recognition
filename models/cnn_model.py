import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """卷积神经网络"""
    def __init__(self):
        super(CNN, self).__init__()
        # 网络结构：
        # 卷积层1 → 池化层1 → 卷积层2 → 池化层2 → 全连接层1 → 全连接层2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 卷积层1：1→32通道，3x3卷积核
        self.pool1 = nn.MaxPool2d(2, 2)                          # 池化层1：2x2最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 卷积层2：32→64通道
        self.pool2 = nn.MaxPool2d(2, 2)                          # 池化层2：2x2最大池化
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                    # 全连接层1：64*7*7→128
        self.fc2 = nn.Linear(128, 10)                            # 输出层：128→10
        self.dropout = nn.Dropout(0.25)                          # 防止过拟合

    def forward(self, x):
        # 卷积+池化1：(batch,1,28,28) → (batch,32,28,28) → (batch,32,14,14)
        x = self.pool1(F.relu(self.conv1(x)))
        # 卷积+池化2：(batch,32,14,14) → (batch,64,14,14) → (batch,64,7,7)
        x = self.pool2(F.relu(self.conv2(x)))
        # 展平：(batch,64,7,7) → (batch,64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_params_count(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)