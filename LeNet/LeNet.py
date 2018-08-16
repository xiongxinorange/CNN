#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.transforms as transforms

# hyper parameters
EPOCH = 20
BATCH_SIZE = 8
LR = 0.0010
DOWNLOAD_CIFAR10 = False

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


if not(os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
    DOWNLOAD_CIFAR10 = True
train_data = torchvision.datasets.CIFAR10(
    root='./cifar10/',
    train=True,
    transform=transform,
    download=DOWNLOAD_CIFAR10
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10/',
    train=False,
    transform=transform,
    download=DOWNLOAD_CIFAR10
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnnnet = LeNet()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnnnet.parameters(), lr=LR, momentum=0.9)

for epoch in range(EPOCH):
    for step, (images, labels) in enumerate(train_loader):
        output = cnnnet(images)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 1999:
            correct, total = .0, .0
            for x, y in test_loader:
                py = cnnnet(x)
                _, predicted = torch.max(py, 1)
                total += y.size(0)
                correct += (predicted == y).sum()
                acc = float(correct) / total

            print('Epoch:', epoch+1, '|Step:', step+1, '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % acc)
