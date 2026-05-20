"""
训练逻辑

包含完整的训练循环和评估函数。
支持：
  - 逐 epoch 训练
  - 训练集/验证集准确率计算
  - 自动保存最佳模型
  - 学习率调度（可选）
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch

    Args:
        model:     神经网络模型
        loader:    训练集 DataLoader
        criterion: 损失函数
        optimizer: 优化器
        device:    计算设备 (cpu/cuda/mps)

    Returns:
        (avg_loss, accuracy): 平均损失和准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # 1. 将数据移动到指定设备
        images = images.to(device)
        labels = labels.to(device)

        # 2. 清零梯度（避免梯度累积）
        optimizer.zero_grad()

        # 3. 前向传播: 计算预测
        outputs = model(images)

        # 4. 计算损失
        loss = criterion(outputs, labels)

        # 5. 反向传播: 计算梯度
        loss.backward()

        # 6. 更新参数
        optimizer.step()

        # === 统计 ===
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()  # 装饰器: 禁用梯度计算（节省显存和计算）
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    评估模型（不计算梯度）

    Args:
        model:     神经网络模型
        loader:    测试集 DataLoader
        criterion: 损失函数
        device:    计算设备

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()  # 切换到评估模式（关闭 Dropout、BatchNorm 等）
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints",
) -> dict:
    """
    完整训练流程

    Args:
        model:        神经网络模型
        train_loader: 训练集 DataLoader
        test_loader:  测试集 DataLoader
        epochs:       训练轮数
        lr:           学习率
        device:       计算设备
        save_dir:     模型保存目录

    Returns:
        history: 训练历史记录
            {
                "train_loss": [...],
                "train_acc":  [...],
                "test_loss":  [...],
                "test_acc":   [...],
            }
    """
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器: 每 5 个 epoch 学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 记录训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0.0
    total_start = time.time()

    print(f"训练设备: {device}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 在测试集上评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} "
            f"| {test_loss:>9.4f} | {test_acc:>8.4f} | {epoch_time:>5.1f}s"
        )

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = save_path / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  -> 保存最佳模型 (acc={best_acc:.4f})")

    total_time = time.time() - total_start
    print("-" * 70)
    print(f"训练完成! 总耗时: {total_time:.1f}s, 最佳测试准确率: {best_acc:.4f}")

    return history
