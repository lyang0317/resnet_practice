"""
ResNet vs PlainCNN 对比实验
数据集：CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision.datasets import ImageFolder

from model import ResNet18, PlainCNN, Plain18, Plain50, ResNet50, Plain50_Bottleneck

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 超参数
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),      # 随机裁剪到 64x64
    transforms.RandomHorizontalFlip(),     # 随机翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # ImageNet 的标准化参数
])

val_transform = transforms.Compose([
    transforms.Resize(64),                 # 验证集统一缩放
    transforms.CenterCrop(64),             # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据
train_dataset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_transform)
test_dataset = ImageFolder(root='./data/tiny-imagenet-200/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'Loss': f'{running_loss/len(train_loader):.3f}',
                          'Acc': f'{100.*correct/total:.2f}%'})

    return 100. * correct / total, running_loss / len(train_loader)


def evaluate(model, test_loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def train_model(model, name, save_path=None):
    """完整训练流程"""
    print(f'\n{"="*50}')
    print(f'训练模型: {name}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'{"="*50}')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 学习率调度
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    train_accs = []
    test_accs = []
    train_losses = []

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        test_acc = evaluate(model, test_loader)
        scheduler.step()

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)

        print(f'Epoch {epoch:2d}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)

    print(f'\n{name} 最佳测试准确率: {best_acc:.2f}%')

    return {
        'name': name,
        'best_acc': best_acc,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses
    }


def plot_comparison(results):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 准确率曲线
    for r in results:
        axes[0].plot(r['test_accs'], label=f"{r['name']} (Best: {r['best_acc']:.1f}%)")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy Compare')
    axes[0].legend()
    axes[0].grid(True)

    # 训练损失曲线
    for r in results:
        axes[1].plot(r['train_losses'], label=r['name'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Training Loss Compare')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('comparison_tiny_imagenet.png', dpi=150)
    plt.show()
    print("对比图已保存为 comparison_tiny_imagenet.png")


if __name__ == '__main__':
    # 创建保存目录
    os.makedirs('checkpoints_tiny_imagenet', exist_ok=True)

    # 运行对比实验
    results = []

    # ========== 新增18层对比 ==========
    print("\n" + "="*60)
    print("开始18层深度对比实验")
    print("="*60)

    # 实验1：普通CNN（无残差连接）
    plain_cnn = PlainCNN(num_classes=200)
    results.append(train_model(plain_cnn, 'PlainCNN (NO Resnet)', 'checkpoints_tiny_imagenet/plain_cnn.pth'))

    # 实验4: ReLU + BN + 4层
    plain18 = Plain18(use_bn=True, activation='relu', num_blocks=1, num_classes=200)
    results.append(train_model(plain18, 'Plain18 (NO Resnet, 4 layers)', 'checkpoints_tiny_imagenet/plain4.pth'))

    # 实验3: ReLU + BN
    plain18 = Plain18(use_bn=False, activation='sigmoid', num_blocks=4, num_classes=200)
    results.append(train_model(plain18, 'Plain18 (NO Resnet, 18 layers, not bn, sigmoid)', 'checkpoints_tiny_imagenet/plain18.pth'))

    # 实验3: ReLU + BN
    plain18 = Plain18(use_bn=False, activation='relu', num_blocks=4, num_classes=200)
    results.append(train_model(plain18, 'Plain18 (NO Resnet, 18 layers, not bn, relu)', 'checkpoints_tiny_imagenet/plain18.pth'))

    # 实验3: ReLU + BN
    plain18 = Plain18(use_bn=True, activation='relu', num_blocks=4, num_classes=200)
    results.append(train_model(plain18, 'Plain18 (NO Resnet, 18 layers)', 'checkpoints_tiny_imagenet/plain18.pth'))

    # 实验2：ResNet-18（有残差连接）
    resnet18 = ResNet18(num_classes=200)
    results.append(train_model(resnet18, 'ResNet18 (Resnet)', 'checkpoints_tiny_imagenet/resnet18.pth'))

    # # 实验3: Sigmoid + 无BN
    # model = Plain18(use_bn=False, activation='sigmoid')
    # results.append(train_model(plain18, 'Plain18 (无残差, 18层)', 'checkpoints/plain18.pth'))

    # ========== 新增50层对比 ==========
    print("\n" + "="*60)
    print("开始50层深度对比实验")
    print("="*60)

    plain50 = Plain50(num_classes=200)
    results.append(train_model(plain50, 'Plain50 (NO Resnet, 50layers)', 'checkpoints_tiny_imagenet/plain50.pth'))

    plain50_Bottleneck = Plain50_Bottleneck(num_classes=200)
    results.append(train_model(plain50_Bottleneck, 'Plain50_Bottleneck (NO Resnet, like bottle block, 50layers)', 'checkpoints_tiny_imagenet/plain50_Bottleneck.pth'))

    resnet50 = ResNet50(num_classes=200)
    results.append(train_model(resnet50, 'ResNet50 (Resnet, 50 layers', 'checkpoints_tiny_imagenet/resnet50.pth'))

    # 绘制对比图
    plot_comparison(results)

    # 输出最终对比表格
    print('\n' + '='*60)
    print('最终对比结果')
    print('='*60)
    print(f"{'模型':<25} {'最佳测试准确率':<20} {'参数量':<15}")
    print('-'*60)
    for r in results:
        params = sum(p.numel() for p in eval(r['name'].replace('(', '_').replace(')', '_').split()[0] + '()').parameters())
        print(f"{r['name']:<25} {r['best_acc']:.2f}%{'':<15} {params:,}")
    print('='*60)