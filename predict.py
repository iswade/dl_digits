"""
预测与推理

支持两种模式:
  1. 对 MNIST 测试集批量预测
  2. 对用户输入的图片进行单张推理
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from model import MNISTCNN
from dataset import get_dataloaders


def load_model(
    model_path: str = "./checkpoints/best_model.pth",
    device: torch.device = torch.device("cpu"),
) -> MNISTCNN:
    """
    加载训练好的模型

    Args:
        model_path: 模型权重文件路径
        device:     计算设备

    Returns:
        加载了权重的模型
    """
    model = MNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: MNISTCNN,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
) -> dict:
    """
    对测试集进行预测并返回结果

    Args:
        model:       训练好的模型
        loader:      测试集 DataLoader
        device:      计算设备
        num_samples: 要返回的样本数量

    Returns:
        {
            "images":     Tensor  (num_samples, 1, 28, 28),
            "true_labels": list    (num_samples,),
            "pred_labels": list    (num_samples,),
            "probabilities": Tensor (num_samples, 10),
        }
    """
    images_list = []
    true_labels = []
    pred_labels = []
    probabilities_list = []

    count = 0
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        batch_size = images.size(0)
        for i in range(batch_size):
            if count >= num_samples:
                break
            images_list.append(images[i].cpu())
            true_labels.append(labels[i].item())
            pred_labels.append(predicted[i].item())
            probabilities_list.append(probs[i].cpu())
            count += 1

        if count >= num_samples:
            break

    return {
        "images": torch.stack(images_list),
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "probabilities": torch.stack(probabilities_list),
    }


def show_prediction(result: dict, num: int = 10):
    """
    打印预测结果。

    如需可视化图像，请使用 utils.py 中的 plot_predictions()。

    Args:
        result: predict() 返回的结果字典
        num:    要展示的样本数
    """
    correct = 0
    for i in range(min(num, len(result["true_labels"]))):
        true_lbl = result["true_labels"][i]
        pred_lbl = result["pred_labels"][i]
        probs = result["probabilities"][i]
        confidence = probs[pred_lbl].item()

        mark = "✓" if true_lbl == pred_lbl else "✗"
        if true_lbl == pred_lbl:
            correct += 1

        print(f"  [{i}] 真实={true_lbl}  预测={pred_lbl}  "
              f"置信度={confidence:.2%}  {mark}")

    acc = correct / min(num, len(result["true_labels"]))
    print(f"\n预测准确率: {acc:.2%} ({correct}/{min(num, len(result['true_labels']))})")


def preprocess_image(
    image_path: str,
    size: int = 28,
) -> torch.Tensor:
    """
    加载并预处理用户输入的图片，使其符合 MNIST 模型输入格式。

    处理流程:
      1. 转为灰度图
      2. 二值化，分离前景（数字）和背景
      3. 找到数字的包围盒，裁剪空白边缘
      4. 保持宽高比缩放到 20x20
      5. 居中放置在 28x28 画布上
      6. 自动反相: 保证数字为白色（255），背景为黑色（0）
      7. 用 MNIST 统计量标准化

    Args:
        image_path: 图片文件路径 (.png/.jpg/.jpeg/.bmp)
        size:       输出图片尺寸（默认 28）

    Returns:
        预处理后的张量，形状 (1, 1, 28, 28)
    """
    # 1. 打开图片，转为灰度
    img = Image.open(image_path).convert("L")

    # 2. 二值化: 阈值 128，纯黑白色
    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)

    # 3. 找出数字的包围盒
    import numpy as np
    arr = np.array(img)
    rows = np.any(arr < 255, axis=1)   # 像素值 <255 的是数字部分
    cols = np.any(arr < 255, axis=0)

    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # 向外扩 2px 留点边
        y_min = max(0, y_min - 2)
        y_max = min(arr.shape[0] - 1, y_max + 2)
        x_min = max(0, x_min - 2)
        x_max = min(arr.shape[1] - 1, x_max + 2)
        img = img.crop((x_min, y_min, x_max + 1, y_max + 1))

    # 4. 保持宽高比缩放到 20x20
    target_size = size - 8  # 20
    w, h = img.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 5. 居中放置在 28x28 画布上
    canvas = Image.new("L", (size, size), 0)  # 黑色背景
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas.paste(img, (x_offset, y_offset))

    # 6. 转 Tensor 并反相: MNIST 是白字黑底
    tensor = torch.from_numpy(np.array(canvas)).float()
    # 如果背景偏白（均值 > 127），就反相
    if tensor.mean() > 127:
        tensor = 255 - tensor

    # 7. 用 MNIST 统计量标准化
    tensor = tensor / 255.0
    tensor = (tensor - 0.1307) / 0.3081

    # 添加 batch 和 channel 维度 -> (1, 1, 28, 28)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


@torch.no_grad()
def infer_image(
    model: MNISTCNN,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    对单张预处理后的图片进行推理

    Args:
        model:        训练好的模型
        image_tensor: 预处理后的图片张量 (1, 1, 28, 28)
        device:       计算设备

    Returns:
        {
            "pred_label":     int,       预测的数字
            "confidence":     float,     预测的置信度
            "probabilities":  Tensor,    10 个类别的概率 (10,)
        }
    """
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1).squeeze(0)  # (10,)

    pred_label = torch.argmax(probs).item()
    confidence = probs[pred_label].item()

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "probabilities": probs.cpu(),
    }


def print_infer_result(result: dict):
    """
    打印单张图片的推理结果

    Args:
        result: infer_image() 返回的结果字典
    """
    probs = result["probabilities"]

    print(f"\n预测结果: {result['pred_label']}")
    print(f"置信度:   {result['confidence']:.2%}")
    print()
    print("各数字概率:")
    for i in range(10):
        bar = "█" * int(probs[i] * 50) + "░" * (50 - int(probs[i] * 50))
        print(f"  {i}: {bar} {probs[i]:.2%}")


if __name__ == "__main__":
    """推理演示"""
    import argparse

    parser = argparse.ArgumentParser(description="MNIST 单图推理")
    parser.add_argument("image", type=str, help="图片文件路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = "./checkpoints/best_model.pth"
    if not Path(model_path).exists():
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先运行 python main.py --mode train 训练模型")
        exit(1)

    model = load_model(model_path, device)
    tensor = preprocess_image(args.image)
    result = infer_image(model, tensor, device)
    print_infer_result(result)
