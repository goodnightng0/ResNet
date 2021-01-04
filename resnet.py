import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time


class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out=self.bn(x)
        out=F.relu(out)
        out=self.conv(out)
        out=self.bn(out)
        out=F.relu(out)
        out=self.conv(out)
        out=out+x
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2,padding=0)
        self.mp=nn.MaxPool2d(2, stride=2,padding=0)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x=self.bn1(x)
        x=F.relu(x)
        out=self.conv1(x)
        x=self.conv3(x)
        out=self.bn2(out)
        out=F.relu(out)
        out=self.conv2(out)
        out=out+x
        return out

class IdentityResNet(nn.Module):

    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512, 10)
        self.stage1 = self.init_stage1()
        self.stage2 = self.init_stage2()
        self.stage3 = self.init_stage3()
        self.stage4 = self.init_stage4()

    def init_stage1(self):
        modules = []
        a=ResidualBlock1(64,64)
        b=ResidualBlock1(64,64)
        a.to(dev);b.to(dev)
        modules.append(a)
        modules.append(b)
        sequential = nn.Sequential(*modules)
        return sequential

    def init_stage2(self):
        modules = []
        a=ResidualBlock2(64, 128)
        b=ResidualBlock1(128, 128)
        a.to(dev);b.to(dev)
        modules.append(a)
        modules.append(b)
        sequential = nn.Sequential(*modules)
        return sequential

    def init_stage3(self):
        modules = []
        a = ResidualBlock2(128, 256)
        b = ResidualBlock1(256, 256)
        a.to(dev);b.to(dev)
        modules.append(a)
        modules.append(b)
        sequential = nn.Sequential(*modules)
        return sequential

    def init_stage4(self):
        modules = []
        a = ResidualBlock2(256, 512)
        b = ResidualBlock1(512, 512)
        a.to(dev);b.to(dev)
        modules.append(a)
        modules.append(b)
        sequential = nn.Sequential(*modules)
        return sequential

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, 4, stride=4,padding=0)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

if torch.cuda.is_available():
    dev=torch.device('cuda:0')
else:
    dev=torch.device('cpu')
print('current device: ', dev)

# data preparation: CIFAR10
# set batch size for training data
batch_size = 8

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),lr=0.005,momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net.forward(inputs)

        # set loss
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end - t_start, ' sec')
            t_start = t_end

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' % (classes[i]), ': ',
          100 * class_correct[i] / class_total[i], '%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct) / sum(class_total)), '%')