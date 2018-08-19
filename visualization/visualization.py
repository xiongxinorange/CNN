#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import visdom

viz = visdom.Visdom()

# Image Preprocessing
to_normalized_tensor = [transforms.ToTensor(), transforms.Normalize((0.5070754, 0.48655024, 0.44091907),
                                                                    (0.26733398, 0.25643876, 0.2761503))]
data_augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

train_data = torchvision.datasets.CIFAR100(root='./cifar100/', train=True, download=True, transform=transforms.Compose(data_augmentation + to_normalized_tensor))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR100(root='./cifar100/', train=False, download=True, transform=transforms.Compose(data_augmentation + to_normalized_tensor))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(SENet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


senet = SENet(ResidualBlock, [3, 8, 36, 3]).cuda()
criterion = nn.CrossEntropyLoss()
lr = 0.0008
optimizer = torch.optim.Adam(senet.parameters(), lr=lr)

lossdata = []
accdata = []
loss = 0
accuracy = 0

for epoch in range(40):
    for step, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = senet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 99:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.cuda())
                outputs = senet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum()
                acc = float(correct) / total
            print('Epoch:', epoch+1, '|Step:', step+1, '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % acc)
    lossdata.append(loss)
    accdata.append(accuracy)

torch.save(senet.state_dict(), './SeResnet.pkl/')
lossdata = torch.Tensor(lossdata)
accdata = torch.Tensor(accdata)
x = torch.range(1, 40)

viz.line(lossdata, x, opts=dict(title='loss'))
viz.line(accdata, x, opts=dict(title='acc'))
