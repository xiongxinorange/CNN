#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
train_data = torchvision.datasets.CIFAR10(
    root='./cifar10/',
    train=True,
    transform=transform,
    download=True
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10/',
    train=False,
    transform=transform,
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')

vggnet = VGG11().cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(vggnet.parameters(), lr=lr)

for epoch in range(10):
    for step, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = vggnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 99:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.cuda())
                outputs = vggnet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum()
                acc = float(correct) / total
            print('Epoch:', epoch+1, '|Step:', step+1, '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % acc)
