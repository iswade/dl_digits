#!/usr/bin/env python3
"""
MNIST 手写数字识别 —— 主入口

统一管理训练、预测、单图推理和可视化的入口脚本。

用法:
  # 训练模型
  python main.py --mode train --epochs 10

  # 在测试集上预测并可视化
  python main.py --mode predict --num 20

  # 仅展示预测错误的样本
  python main.py --mode errors

  # 对用户输入的图片进行推理
  python main.py --mode infer --image path/to/digit.png

  # 从测试集中导出一张图片用于测试
  python main.py --mode export --num 3
"""

import argparse
from pathlib import Path

import torch

from model import MNISTCNN, count_parameters
from dataset import get_dataloaders
from train import train
from predict import load_model, predict, show_prediction, preprocess_image, infer_image, print_infer_result
from utils import plot_training_history, plot_predictions, show_misclassified, plot_infer_result


def get_device() -> torch.device:
    """自动选择可用的计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST 手写数字识别")

    # 运行模式
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "predict", "errors", "infer", "export"],
        help="运行模式 (默认: train)",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch 大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")

    # 预测参数
    parser.add_argument(
        "--num", type=int, default=10,
        help="预测/展示的样本数量",
    )

    # 单图推理参数
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="要推理的图片文件路径（用于 infer 模式）",
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="保存推理结果可视化图片",
    )

    # 模型和数据路径
    parser.add_argument(
        "--model-path", type=str, default="./checkpoints/best_model.pth",
        help="模型文件路径",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="数据集目录")

    return parser.parse_args()


def mode_train(args):
    """训练模式"""
    device = get_device()
    print(f"当前设备: {device}")
    print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # 1. 创建模型
    model = MNISTCNN().to(device)
    print(f"模型参数量: {count_parameters(model):,}")

    # 2. 加载数据
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    print(f"训练集 batch 数: {len(train_loader)}")
    print(f"测试集 batch 数: {len(test_loader)}")

    # 3. 开始训练
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # 4. 绘制训练曲线
    plot_training_history(history, save_path="./checkpoints/training_history.png")

    print("\n训练完成！运行以下命令查看预测效果：")
    print("  python main.py --mode predict --num 20")
    print("  python main.py --mode errors")


def mode_predict(args):
    """预测模式"""
    device = get_device()

    # 检查模型文件是否存在
    if not Path(args.model_path).exists():
        print(f"错误: 未找到模型文件 '{args.model_path}'")
        print("请先运行: python main.py --mode train")
        return

    # 1. 加载模型
    model = load_model(args.model_path, device)
    print(f"模型已加载: {args.model_path}")

    # 2. 加载测试数据
    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    # 3. 预测
    result = predict(model, test_loader, device, num_samples=args.num)

    # 4. 展示结果
    show_prediction(result, num=args.num)
    plot_predictions(result, num=args.num, save_path="./checkpoints/predictions.png")


def mode_errors(args):
    """展示预测错误的样本"""
    device = get_device()

    if not Path(args.model_path).exists():
        print(f"错误: 未找到模型文件 '{args.model_path}'")
        print("请先运行: python main.py --mode train")
        return

    model = load_model(args.model_path, device)
    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    result = predict(model, test_loader, device, num_samples=args.num)
    show_misclassified(result, save_path="./checkpoints/misclassified.png")


def mode_infer(args):
    """单图推理模式"""
    if not args.image:
        print("错误: infer 模式需要指定 --image 参数")
        print("用法: python main.py --mode infer --image path/to/digit.png")
        return

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片文件不存在: {args.image}")
        return

    device = get_device()

    if not Path(args.model_path).exists():
        print(f"错误: 未找到模型文件 '{args.model_path}'")
        print("请先运行: python main.py --mode train")
        return

    # 1. 加载模型
    model = load_model(args.model_path, device)
    print(f"设备:           {device}")
    print(f"模型:           {args.model_path}")
    print(f"输入图片:       {args.image}")

    # 2. 预处理图片
    print("正在预处理图片...", end=" ")
    tensor = preprocess_image(str(image_path))
    print("完成")

    # 3. 推理
    print("正在推理...", end=" ")
    result = infer_image(model, tensor, device)
    print("完成")

    # 4. 打印结果
    print_infer_result(result)

    # 5. 可视化（仅 --save 时保存图片，不弹 GUI 窗口）
    if args.save:
        save_path = "./checkpoints/infer_result.png"
        plot_infer_result(
            image_tensor=tensor,
            result=result,
            image_path=str(image_path),
            save_path=save_path,
        )


def mode_export(args):
    """
    从 MNIST 测试集中导出几张图片，供 infer 模式测试用。

    导出的图片会保存在 ./test_images/ 目录下，
    文件名为 mnist_{真实标签}_{索引}.png
    """
    from PIL import Image as PILImage
    import numpy as np

    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    export_dir = Path("./test_images")
    export_dir.mkdir(exist_ok=True)

    count = 0
    for images, labels in test_loader:
        for i in range(images.size(0)):
            if count >= args.num:
                break

            img = images[i].squeeze().numpy()
            # 反标准化回 [0, 1]
            img = img * 0.3081 + 0.1307
            # 转为 0-255 灰度图
            img = (img * 255).astype(np.uint8)

            label = labels[i].item()
            filepath = export_dir / f"mnist_{label}_{count}.png"
            PILImage.fromarray(img).save(filepath)
            print(f"  已导出: {filepath}")
            count += 1

        if count >= args.num:
            break

    print(f"\n共导出 {count} 张图片到 {export_dir}/")
    print("使用示例: python main.py --mode infer --image test_images/mnist_3_0.png")


def main():
    args = parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "predict":
        mode_predict(args)
    elif args.mode == "errors":
        mode_errors(args)
    elif args.mode == "infer":
        mode_infer(args)
    elif args.mode == "export":
        mode_export(args)


if __name__ == "__main__":
    main()
