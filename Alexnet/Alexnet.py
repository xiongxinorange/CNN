from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
import torchvision
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


NUM_CLASSES = 10
class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

alexnet = AlexNet().cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(alexnet.parameters(), lr=lr)

for epoch in range(20):
    for step, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 99:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.cuda())
                outputs = alexnet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum()
                acc = float(correct) / total
            print('Epoch:', epoch+1, '|Step:', step+1, '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % acc)
