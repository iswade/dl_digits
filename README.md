# MNIST 手写数字识别 - PyTorch 学习项目

一个结构清晰的 PyTorch 入门项目，用于识别手写数字（0-9）。

## 项目结构

```
mnist_learn/
├── model.py      # 神经网络模型定义 (CNN)
├── dataset.py    # 数据加载与预处理
├── train.py      # 训练逻辑
├── predict.py    # 预测/推理（含单图推理）
├── utils.py      # 工具函数（可视化）
├── main.py       # 统一入口
├── requirements.txt
├── checkpoints/  # 训练输出（模型权重 + 可视化图片）
└── test_images/  # 导出的测试图片
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（10 个 epoch，准确率 ~99%）
python main.py --mode train --epochs 10

# 3. 在测试集上预测并可视化
python main.py --mode predict --num 20

# 4. 从测试集导出几张图片，用于推理测试
python main.py --mode export --num 5

# 5. 对导出的图片进行推理
python main.py --mode infer --image test_images/mnist_7_0.png --save
```

## 全部功能

### train — 训练模型

```bash
python main.py --mode train --epochs 10 --batch-size 64 --lr 0.001
```

训练完成后自动保存最佳模型到 `checkpoints/best_model.pth`，并绘制训练曲线。

### predict — 测试集批量预测

```bash
python main.py --mode predict --num 20
```

随机展示测试集中的图片，显示预测结果和置信度，绿色=正确，红色=错误。

### infer — 对任意图片推理（核心功能）

```bash
# 推理单张图片
python main.py --mode infer --image path/to/digit.png

# 推理并保存可视化结果
python main.py --mode infer --image path/to/digit.png --save
```

支持常见的图片格式（PNG、JPG、JPEG、BMP）。程序会自动：
1. 转为灰度图
2. 二值化并裁剪空白边缘
3. 缩放到 20x20 并居中放置在 28x28 画布上
4. 自动检测并反相（保证数字为白色，背景为黑色）
5. 用 MNIST 统计量标准化

**如何获取测试图片：**
- 运行 `python main.py --mode export --num 5` 从测试集导出
- 用系统自带的「预览」或「画图」涂鸦一个数字，保存为 PNG
- 用手机拍一张手写数字照片

### errors — 分析预测失败的样本

```bash
python main.py --mode errors
```

专门展示模型预测错误的样本，方便分析模型弱点。

### export — 导出测试图片

```bash
python main.py --mode export --num 10
```

从 MNIST 测试集中导出图片到 `test_images/` 目录，用于 infer 模式的测试。

## 学习路线

建议按文件依赖顺序阅读：

1. **model.py** — 了解 CNN 的层结构（Conv2D、Pooling、FC、Dropout）
2. **dataset.py** — 学习数据加载与增强（Transform、DataLoader）
3. **train.py** — 掌握训练循环的完整流程（前向、反向、优化、评估）
4. **predict.py** — 学会模型推理（批量预测 + 单图推理 + 图片预处理）
5. **utils.py** — 可视化工具（训练曲线、预测结果、概率柱状图）
6. **main.py** — 理解各模块如何通过 CLI 参数串联

## 单图推理流程

```
用户输入图片
    ↓
灰度化 + 二值化       preprocess_image()
    ↓
裁剪空白 + 缩放到 20x20
    ↓
居中放置到 28x28 画布
    ↓
自动反相（白字黑底）
    ↓
标准化 (mean=0.1307, std=0.3081)
    ↓
CNN 模型推理           infer_image()
    ↓
输出: 数字 + 置信度 + 10 类概率
    ↓
可视化: 图片 + 概率柱状图  plot_infer_result()
```
