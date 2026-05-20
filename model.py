"""
神经网络模型定义 (CNN for MNIST)

网络结构:
  Input (28x28 grayscale)
    -> Conv1 (32ch, 3x3) + ReLU + MaxPool(2x2)
    -> Conv2 (64ch, 3x3) + ReLU + MaxPool(2x2)
    -> Flatten
    -> FC1 (128) + ReLU + Dropout(0.5)
    -> FC2 (10) + Softmax
    -> Output (10 classes)

输入:  (batch_size, 1, 28, 28)  灰度图
输出:  (batch_size, 10)         10 个类别的概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """用于 MNIST 的卷积神经网络"""

    def __init__(self):
        super().__init__()

        # === 卷积层 ===
        # 第一层: 1通道 -> 32通道，提取低级特征（边缘、线条）
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 输入通道数（灰度图为1）
            out_channels=32,  # 输出通道数（卷积核数量）
            kernel_size=3,    # 卷积核大小 3x3
            padding=1,        # 保持尺寸不变
        )
        # 第二层: 32通道 -> 64通道，提取高级特征（形状、图案）
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        # 池化层（共用，不包含可学习参数）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # === 全连接层 ===
        # 经过两次池化后，28x28 变为 7x7
        # 展平后维度: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, 1, 28, 28)

        Returns:
            输出张量，形状 (batch_size, 10)
        """
        # 卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64, 7, 7)

        # 展平，保留 batch 维度
        x = x.view(x.size(0), -1)              # (B, 3136)

        # 全连接 -> 激活 -> Dropout
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.dropout(x)

        # 输出层（返回 logits，CrossEntropyLoss 内部做 Softmax）
        x = self.fc2(x)                        # (B, 10)

        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型的可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """测试模型结构"""
    model = MNISTCNN()
    dummy = torch.randn(4, 1, 28, 28)  # 模拟一个 batch 的输入
    output = model(dummy)

    print(f"模型参数量: {count_parameters(model):,}")
    print(f"输入形状:  {dummy.shape}")
    print(f"输出形状:  {output.shape}")
    print(f"输出概率:  {F.softmax(output, dim=1)[0].detach().numpy()}")
