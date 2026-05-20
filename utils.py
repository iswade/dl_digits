"""
工具函数 —— 可视化训练过程和预测结果
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体（macOS 使用 PingFang SC）
for font_name in ["PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False  # 修复负号显示


def plot_training_history(history: dict, save_path: str = None):
    """
    绘制训练过程的损失和准确率曲线

    Args:
        history:  train() 返回的历史记录字典
        save_path: 图片保存路径，为 None 则直接显示
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左图: 损失曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="训练损失")
    ax1.plot(epochs, history["test_loss"], "r-", label="测试损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 准确率曲线
    ax2.plot(epochs, history["train_acc"], "b-", label="训练准确率")
    ax2.plot(epochs, history["test_acc"], "r-", label="测试准确率")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    else:
        plt.show()


def plot_predictions(result: dict, num: int = 10, save_path: str = None):
    """
    可视化预测结果: 显示图像 + 真实标签 + 预测标签 + 置信度

    Args:
        result:    predict() 返回的结果字典
        num:       要展示的样本数
        save_path: 图片保存路径
    """
    images = result["images"]
    true_labels = result["true_labels"]
    pred_labels = result["pred_labels"]
    probabilities = result["probabilities"]

    num = min(num, len(images))
    cols = 5
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i in range(num):
        ax = axes[i]

        # 图像: 反标准化并显示
        img = images[i].squeeze().numpy()
        # MNIST 标准化: (x * 0.3081) + 0.1307
        img = img * 0.3081 + 0.1307
        ax.imshow(img, cmap="gray")

        true_lbl = true_labels[i]
        pred_lbl = pred_labels[i]
        confidence = probabilities[i][pred_lbl].item()

        # 颜色: 正确=绿色, 错误=红色
        color = "green" if true_lbl == pred_lbl else "red"
        title = f"真实: {true_lbl}\n预测: {pred_lbl} ({confidence:.1%})"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis("off")

    # 隐藏多余的子图
    for i in range(num, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle("MNIST 预测结果（绿色=正确，红色=错误）", y=1.02, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()


def plot_infer_result(
    image_tensor: torch.Tensor,
    result: dict,
    image_path: str = None,
    save_path: str = None,
):
    """
    可视化单张图片的推理结果: 输入图片 + 概率柱状图

    Args:
        image_tensor: 预处理后的图片张量 (1, 1, 28, 28)
        result:       infer_image() 返回的结果字典
        image_path:   原始图片路径（用于标题显示）
        save_path:    图片保存路径
    """
    pred_label = result["pred_label"]
    confidence = result["confidence"]
    probs = result["probabilities"].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [1, 2]})

    # === 左图: 输入图片 ===
    img = image_tensor.squeeze().numpy()
    img = img * 0.3081 + 0.1307  # 反标准化
    img = np.clip(img, 0, 1)

    ax1.imshow(img, cmap="gray")
    title = f"预测: {pred_label}" if image_path is None else \
            f"预测: {pred_label}\n({Path(image_path).name})"
    ax1.set_title(title, fontsize=14, pad=10)
    ax1.axis("off")

    # === 右图: 概率柱状图 ===
    colors = ["#e74c3c"] * 10
    colors[pred_label] = "#2ecc71"  # 预测结果用绿色高亮

    bars = ax2.barh(range(10), probs, color=colors, height=0.6)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Digit")
    ax2.set_title(f"预测 = {pred_label}  (置信度: {confidence:.2%})",
                  fontsize=13, color="#2ecc71" if confidence > 0.5 else "#e74c3c")
    ax2.set_xlim(0, 1.0)
    ax2.set_yticks(range(10))
    ax2.invert_yaxis()  # 0 在上面
    ax2.grid(axis="x", alpha=0.3)

    # 在柱状条上标注百分比
    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax2.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.suptitle("MNIST 单图推理结果", fontsize=15, y=1.03)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()


def show_misclassified(result: dict, save_path: str = None):
    """
    专门展示预测失败的样本，便于分析模型弱点

    Args:
        result:    predict() 返回的结果字典
        save_path: 图片保存路径
    """
    images = result["images"]
    true_labels = result["true_labels"]
    pred_labels = result["pred_labels"]

    # 找出预测错误的样本
    mis_indices = [
        i for i in range(len(true_labels))
        if true_labels[i] != pred_labels[i]
    ]

    if not mis_indices:
        print("没有预测错误的样本！")
        return

    num = min(len(mis_indices), 10)
    cols = 5
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i in range(num):
        ax = axes[i]
        idx = mis_indices[i]
        img = images[idx].squeeze().numpy()
        img = img * 0.3081 + 0.1307

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"真实: {true_labels[idx]} → 预测: {pred_labels[idx]}",
            color="red",
            fontsize=10,
        )
        ax.axis("off")

    for i in range(num, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle(f"预测失败的样本（共 {len(mis_indices)} 个）", y=1.02, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
