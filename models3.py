import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import math
import os


class ConvNet(nn.Module):

    """  7 Convloutions Layer and 2 Fully-Connected Layer. """

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)

        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=2)
        self.bn6 = nn.BatchNorm2d(num_features=128)

        self.conv7 = nn.Conv2d(in_channels=128,out_channels=10,kernel_size=3,padding=2)
        self.bn7 = nn.BatchNorm2d(num_features=10)

        self.fc1 = nn.Linear(in_features=10 * 6 * 6, out_features=10)

        self.fc2 = nn.Linear(in_features=10, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.bm1 = nn.BatchNorm2d(num_features=64)
        self.bm2 = nn.BatchNorm2d(num_features=128)
        self.bm3 = nn.BatchNorm2d(num_features=10)

        self.fn1 = nn.BatchNorm2d(num_features=10)
        self.fn2 = nn.BatchNorm2d(num_features=num_classes)


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     ##   he_normal()

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m,nn.Linear):                                                  ##  identity
                nn.init.eye_(m.weight)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.max_pool2d(x, 2,2)
        x = self.bm1(x)

        # print(x.size())

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = F.max_pool2d(x, 2)
        x = self.bm2(x)

        x = F.relu(self.bn7(self.conv7(x)))

        x = F.avg_pool2d(x,2,2)
        x = self.bm3(x)

        # print(x.size())

        x = x.view(-1, 10 * 6 * 6)

        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))


        y = self.softmax(x)
        return x, y


__factory = {
    'cnn': ConvNet,
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

