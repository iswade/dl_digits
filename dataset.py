"""
数据加载与预处理

使用 torchvision 自动下载 MNIST 数据集，
并提供训练集和测试集的 DataLoader。
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """
    定义数据预处理流程

    Return:
        train_transform: 训练集变换（含数据增强）
        test_transform:  测试集变换（仅标准化）
    """
    # 训练集: 水平翻转 + 旋转 + 转Tensor + 标准化
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,              # 随机旋转 ±10 度
            translate=(0.1, 0.1),    # 随机平移 ±10%
        ),
        transforms.ToTensor(),       # PIL -> Tensor，像素值归一化到 [0, 1]
        transforms.Normalize(
            mean=(0.1307,),          # MNIST 数据集像素均值
            std=(0.3081,),           # MNIST 数据集像素标准差
        ),
    ])

    # 测试集: 仅转 Tensor 和标准化（不增强，保证评估稳定）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    return train_transform, test_transform


def get_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    创建训练集和测试集的 DataLoader

    Args:
        batch_size:  每个 batch 的样本数
        data_dir:    数据存放目录
        num_workers: 数据加载的并行进程数

    Returns:
        (train_loader, test_loader)
    """
    train_transform, test_transform = get_transforms()

    # 下载/加载 MNIST 训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # 下载/加载 MNIST 测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 训练集打乱，增加随机性
        num_workers=num_workers,
        pin_memory=True,    # 加速 GPU 数据传输
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,      # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    """测试数据加载"""
    train_loader, test_loader = get_dataloaders(batch_size=16)

    # 查看一个 batch 的数据
    images, labels = next(iter(train_loader))
    print(f"训练集总 batch 数: {len(train_loader)}")
    print(f"测试集总 batch 数: {len(test_loader)}")
    print(f"Batch 图像形状:   {images.shape}")  # (16, 1, 28, 28)
    print(f"Batch 标签形状:   {labels.shape}")   # (16,)
    print(f"标签范围:         {labels.min().item()} ~ {labels.max().item()}")
