"""
从零实现ResNet-18
核心组件：残差块、BatchNorm、全局平均池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet基本残差块（用于ResNet-18/34）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主路径：两个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：如果需要维度匹配，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 跳跃连接

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差相加
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    """ResNet主类"""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 四个残差阶段
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class PlainCNN(nn.Module):
    """无残差连接的普通CNN（用于对比实验）"""

    def __init__(self, num_classes=10):
        super(PlainCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Plain18(nn.Module):
    """18层普通CNN，结构与ResNet-18类似但无残差连接"""

    def __init__(self, use_bn=True, activation='relu', num_blocks=4, num_classes=10):
        super(Plain18, self).__init__()

        self.use_bn = use_bn
        self.activation = activation

        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # 初始卷积层（和ResNet一样）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()

        # 4个阶段，每个阶段4层卷积（共16层）+ 初始1层 + 全连接1层 = 18层
        # 每个阶段内的卷积层数 = num_blocks × 2
        self.layer1 = self._make_layer(64, 64, num_blocks, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks, stride=2)

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """创建连续的卷积层组（无残差）"""
        layers = []

        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            s = stride if i == 0 else 1

            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3,
                                    stride=s, padding=1, bias=not self.use_bn))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self.act)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



# ========== Plain-50：50层普通卷积网络 ==========
class Plain50(nn.Module):
    """50层普通CNN，无残差连接"""
    def __init__(self, use_bn=True, activation='relu', num_classes=10):
        super(Plain50, self).__init__()
        self.use_bn = use_bn

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        # 初始卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()

        # 50层 = 初始1层 + 4个阶段 + 全连接1层
        # 每个阶段需要：(50 - 1 - 1) / 4 = 12 层卷积 per stage
        self.layer1 = self._make_layer(64, 64, num_blocks=12, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=12, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=12, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=12, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            s = stride if i == 0 else 1
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3,
                                    stride=s, padding=1, bias=not self.use_bn))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self.act)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ========== ResNet-50：使用瓶颈块 ==========
class Bottleneck(nn.Module):
    """ResNet-50 的瓶颈块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50"""
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet-50: [3, 4, 6, 3] 个瓶颈块
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)  # Bottleneck.expansion = 4

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class PlainBottleneck(nn.Module):
    """无残差连接的瓶颈块（用于Plain50）"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 注意：没有残差相加！只有激活
        out = self.act(out)
        return out


class Plain50_Bottleneck(nn.Module):
    """基于Bottleneck的Plain50（公平对比ResNet50）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 和ResNet50相同的结构，但每块没有残差
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PlainBottleneck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out