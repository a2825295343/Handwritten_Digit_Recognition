import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from .cnn_model import CNN
from .mlp_model import MLP

# 设备配置：优先使用GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
def load_mnist_data(batch_size=64):
    """加载MNIST手写数字数据集"""
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 模型训练函数（新增训练过程数据记录）
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """训练模型并记录训练过程数据"""
    model.to(DEVICE)
    model.train()
    start_time = time.time()

    # 初始化训练过程记录
    train_history = {
        'epoch': [],          # 训练轮次
        'loss': [],           # 每轮平均损失
        'train_accuracy': [], # 每轮训练集准确率
        'time_per_epoch': [], # 每轮耗时（秒）
        'params_count': model.get_params_count() / 10000  # 模型参数量（万）
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向传播+反向传播+优化
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练集准确率（每批）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        # 记录每轮的训练数据
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start

        train_history['epoch'].append(epoch + 1)
        train_history['loss'].append(epoch_loss)
        train_history['train_accuracy'].append(epoch_accuracy)
        train_history['time_per_epoch'].append(epoch_time)

    total_train_time = time.time() - start_time
    print(f'Training completed in {total_train_time:.2f} seconds')
    return model, train_history

# 模型评估函数
def evaluate_model(model, test_loader):
    """评估模型最终准确率和耗时"""
    model.eval()
    correct = 0
    total = 0
    infer_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 记录单批次推理时间
            start_time = time.time()
            outputs = model(images)
            infer_time = (time.time() - start_time) * 1000  # 转换为毫秒
            infer_times.append(infer_time / len(images))  # 单张图片耗时

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_infer_time = sum(infer_times) / len(infer_times)
    params_count = model.get_params_count() / 10000  # 转换为万

    return {
        'accuracy': accuracy,
        'avg_infer_time': avg_infer_time,
        'params_count': params_count,
        'layers': get_model_layers(model)
    }

# 获取模型层数
def get_model_layers(model):
    """统计模型的有效层数"""
    if isinstance(model, MLP):
        return 3  # 3层全连接
    elif isinstance(model, CNN):
        return 6  # 2卷积+2池化+2全连接
    else:
        return 0

# 初始化并训练模型（首次运行时执行）
def init_models():
    """初始化并训练MLP和CNN模型，返回训练好的模型和训练历史"""
    # 加载数据
    train_loader, test_loader = load_mnist_data()

    # 1. 初始化MLP模型
    mlp = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
    print('Training MLP model...')
    mlp, mlp_history = train_model(mlp, train_loader, criterion, optimizer_mlp, epochs=5)

    # 2. 初始化CNN模型
    cnn = CNN()
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=0.001)
    print('Training CNN model...')
    cnn, cnn_history = train_model(cnn, train_loader, criterion, optimizer_cnn, epochs=5)

    # 保存模型
    if not os.path.exists('./data'):
        os.makedirs('./data')
    torch.save(mlp.state_dict(), './data/mlp_model.pth')
    torch.save(cnn.state_dict(), './data/cnn_model.pth')

    # 评估模型最终指标
    mlp_metrics = evaluate_model(mlp, test_loader)
    cnn_metrics = evaluate_model(cnn, test_loader)

    print('\nModel Final Evaluation Results:')
    print(f'MLP - Accuracy: {mlp_metrics["accuracy"]:.2f}%, Params: {mlp_metrics["params_count"]:.2f}万, Infer Time: {mlp_metrics["avg_infer_time"]:.2f}ms')
    print(f'CNN - Accuracy: {cnn_metrics["accuracy"]:.2f}%, Params: {cnn_metrics["params_count"]:.2f}万, Infer Time: {cnn_metrics["avg_infer_time"]:.2f}ms')

    # 生成训练过程对比曲线图
    plot_training_history(mlp_history, cnn_history)
    print(f'Training history plot saved to ./static/training_history.png')

    return mlp, cnn, mlp_history, cnn_history, mlp_metrics, cnn_metrics

# 生成训练过程对比曲线图
def plot_training_history(mlp_history, cnn_history):
    """生成训练过程对比曲线图（横坐标：训练轮次）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MLP vs CNN 训练过程对比曲线', fontsize=16, fontweight='bold')

    # 1. 训练损失变化曲线
    axes[0,0].plot(mlp_history['epoch'], mlp_history['loss'],
                   marker='o', linewidth=2, color='#e74c3c', label='MLP')
    axes[0,0].plot(cnn_history['epoch'], cnn_history['loss'],
                   marker='s', linewidth=2, color='#2ecc71', label='CNN')
    axes[0,0].set_title('训练损失（Loss）变化')
    axes[0,0].set_xlabel('训练轮次（Epoch）')
    axes[0,0].set_ylabel('损失值（Loss）')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. 训练准确率变化曲线
    axes[0,1].plot(mlp_history['epoch'], mlp_history['train_accuracy'],
                   marker='o', linewidth=2, color='#e74c3c', label='MLP')
    axes[0,1].plot(cnn_history['epoch'], cnn_history['train_accuracy'],
                   marker='s', linewidth=2, color='#2ecc71', label='CNN')
    axes[0,1].set_title('训练准确率（Accuracy）变化')
    axes[0,1].set_xlabel('训练轮次（Epoch）')
    axes[0,1].set_ylabel('准确率（%）')
    axes[0,1].set_ylim(80, 100)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. 每轮训练耗时变化曲线
    axes[1,0].plot(mlp_history['epoch'], mlp_history['time_per_epoch'],
                   marker='o', linewidth=2, color='#e74c3c', label='MLP')
    axes[1,0].plot(cnn_history['epoch'], cnn_history['time_per_epoch'],
                   marker='s', linewidth=2, color='#2ecc71', label='CNN')
    axes[1,0].set_title('每轮训练耗时变化')
    axes[1,0].set_xlabel('训练轮次（Epoch）')
    axes[1,0].set_ylabel('耗时（秒/轮）')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. 参数量对比
    models = ['MLP', 'CNN']
    params = [mlp_history['params_count'], cnn_history['params_count']]
    axes[1,1].bar(models, params, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    axes[1,1].set_title('模型参数量对比（静态）')
    axes[1,1].set_ylabel('参数量（万）')
    # 标注数值
    for x, y in zip(models, params):
        axes[1,1].text(x, y+0.2, f'{y:.1f}', ha='center', va='bottom')
    axes[1,1].grid(True, alpha=0.3, axis='y')

    # 调整布局，保存图片
    plt.tight_layout()
    if not os.path.exists('./static'):
        os.makedirs('./static')
    plt.savefig('./static/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 返回训练历史数据
    df_mlp = pd.DataFrame(mlp_history)
    df_mlp['model'] = 'MLP'
    df_cnn = pd.DataFrame(cnn_history)
    df_cnn['model'] = 'CNN'
    df = pd.concat([df_mlp, df_cnn], ignore_index=True)
    return df

# 加载预训练模型
def load_pretrained_models():
    """加载预训练的MLP和CNN模型"""
    # 初始化模型
    mlp = MLP()
    cnn = CNN()

    # 加载权重（如果不存在则先训练）
    try:
        mlp.load_state_dict(torch.load('./data/mlp_model.pth', map_location=DEVICE))
        cnn.load_state_dict(torch.load('./data/cnn_model.pth', map_location=DEVICE))
        # 补充训练历史
        # train_loader, _ = load_mnist_data()
        # criterion = nn.CrossEntropyLoss()
        # optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
        # optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=0.001)
        # mlp, mlp_history = train_model(mlp, train_loader, criterion, optimizer_mlp, epochs=5)
        # cnn, cnn_history = train_model(cnn, train_loader, criterion, optimizer_cnn, epochs=5)
        # plot_training_history(mlp_history, cnn_history)
    except FileNotFoundError:
        print('Pretrained models not found, training from scratch...')
        mlp, cnn, _, _, _, _ = init_models()

    # 移至设备
    mlp.to(DEVICE)
    cnn.to(DEVICE)
    # 设置为评估模式
    mlp.eval()
    cnn.eval()

    return mlp, cnn

# 预测单张图片
def predict_digit(model, image_data):
    """预测手写数字（输入为28x28灰度数组）"""
    # 转换为tensor并添加维度
    image_tensor = torch.tensor(image_data, dtype=torch.float32).reshape(1, 1, 28, 28)
    image_tensor = (image_tensor - 0.1307) / 0.3081  # 归一化
    image_tensor = image_tensor.to(DEVICE)

    # 记录推理时间
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    infer_time = (time.time() - start_time) * 1000

    return {
        'prediction': int(predicted.item()),
        'infer_time': round(infer_time, 2)
    }