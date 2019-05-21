import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import math
import os
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=128,affine=False)

        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=128,affine=False)

        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=128,affine=False)


        self.conv7 = nn.Conv2d(in_channels=128,out_channels=10,kernel_size=1,padding=0)

        # self.conv7 = nn.Conv2d(in_channels=128,out_channels=10,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=10,affine=False)

        # self.fc1 = nn.Linear(in_features=10 * 3 * 3, out_features=10)
        # self.fc1 = nn.Linear(in_features=10*4*4, out_features=10)
        self.fc1 = nn.Linear(in_features=10 * 1 * 1, out_features=10)

        self.fc2 = nn.Linear(in_features=10, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.bm1 = nn.BatchNorm2d(64,affine=False)
        self.bm2 = nn.BatchNorm2d(128,affine=False)
        self.bm3 = nn.BatchNorm2d(10,affine=False)

        self.fn1 = nn.BatchNorm1d(10,affine=False)
        self.fn2 = nn.BatchNorm1d(num_classes,affine=False)

        self.fn3 = nn.BatchNorm1d(10,affine=False)


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')   ##
                nn.init.constant_(m.bias,0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.constant_(m.bias,0)

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

        # print(x.size())

        x = F.max_pool2d(x, 2)
        x = self.bm2(x)

        x = F.relu(self.bn7(self.conv7(x)))

        x = F.avg_pool2d(x,2,2)
        x = self.bm3(x)

        # print(x.size())

        # x = x.view(-1, 10 * 3 * 3)
        # x = x.view(-1,10*4*4)
        x = x.view(-1,10*1*1)

        x = F.relu(self.fn1(self.fc1(x)))
        x = F.relu(self.fn2(self.fc2(x)))

        # print(x.size())

        # my_max = torch.max(x,dim=1)
        # b = x - ma

        b = torch.exp(x)

        a = self.fn3(x)
        y = self.softmax(x)

        # print("--------------x-----------")
        # print(x)
        # print("--------------y----------")
        # print(y)
        return x, y


__factory = {
    'cnn': ConvNet,
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class LossFunc(nn.Module):
    def __init__(self,my_u,my_l,eps):
        super(LossFunc,self).__init__()
        self.my_u = my_u
        self.my_l = my_l
        self.eps = eps


    def forward(self, b):

        # norm = b.norm(p=2, dim=1, keepdim=True)
        # bn = b.div(norm.expand_as(b))

        bn = l2_norm(b)

        print("----------b------------")
        print(b)
        print("---------------------")

        print("-----bn-----")
        print(bn)
        print("----------")

        sim = torch.mm(bn, bn.t())

        # print("-------sim-----")
        # print(sim)
        # print("------------")

        pos_loc = torch.gt(sim, self.my_u).type(torch.float32)
        neg_loc = torch.lt(sim, self.my_l).type(torch.float32)

        pos_entropy = torch.mul(-torch.log(torch.clamp(sim, self.eps, 1.0)), pos_loc)
        neg_entropy = torch.mul(-torch.log(torch.clamp((1 - sim), self.eps, 1.0)), neg_loc)
        loss_sum = torch.mean(pos_entropy + neg_entropy)

        return loss_sum

class LossFunc2(nn.Module):
    def __init__(self):
        super(LossFunc2,self).__init__()


    def forward(self, b):

        # norm = b.norm(p=2, dim=1, keepdim=True)
        # bn = b.div(norm.expand_as(b))

        bn = l2_norm(b)

        # print("-----bn-----")
        # print(bn)
        # print("----------")
        # bn = b

        # print("----------b------------")
        # print(b)
        # print("---------------------")
        #
        # print("-----bn-----")
        # print(bn)
        # print("----------")


        sim = torch.mm(bn, bn.t())



        # print("-------sim-----")
        # print(sim)
        # print("------------")
        # print("my_u,my_l: ")
        # print(my_u)
        # print(my_l)

        pos_loc = torch.gt(sim, my_u).type(torch.float32)
        neg_loc = torch.lt(sim, my_l).type(torch.float32)

        pos_entropy = torch.mul(-torch.log(torch.clamp(sim, eps, 1.0)), pos_loc)
        neg_entropy = torch.mul(-torch.log(torch.clamp((1 - sim), eps, 1.0)), neg_loc)
        loss_sum = torch.mean(pos_entropy + neg_entropy)

        return loss_sum


def clustering_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.manual_seed(1)
    use_gpu = True
    x = torch.randn(10,1,28,28)
    # print(torch.argmax(x,dim=1))
    # print(x)
    print("-------")
    net = ConvNet(num_classes=10)
    # net.apply(weights_init)
    if use_gpu:
        net = net.cuda()

    # weights_init(net)
    # a,b = net(x)
    # print(b)
    #
    # c = torch.argmax(b,dim=1)
    # print(c)
    #
    # print(a.size())
    # print(b.size())
    #
    # print("-------b---------")
    #
    # # print(b)
    # print(torch.argmax(b,dim=1))
    #
    # bn1 = F.normalize(b,p=2,dim=1)
    # # print(bn1)
    # sim1 = torch.mm(bn1,bn1.t())
    # print(sim1)
    # print("------------------")
    #
    # # bn2 = F.normalize(b,p=2,dim=1)
    # bn2 = l2_norm(b)
    # sim2 = torch.mm(bn2,bn2.t())
    # print(sim2)
    # print("-----------")
    #
    #
    # norm  = b.norm(p=2,dim=1,keepdim=True)
    # bn = b.div(norm.expand_as(b))
    #
    # print("----------")
    # print(bn)
    # print("----------")
    #
    # sim = torch.mm(bn,bn.t())
    #
    # print(sim)
    # torchvision.datasets.MNIST
    from torchvision import transforms
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((,), (0.3081,))
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((,), (0.3081,))
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=512, shuffle=False)


    optimizer_model = torch.optim.RMSprop(net.parameters(), lr=0.001)

    my_u = 0.95
    # my_u = 0.5
    my_l = 0.455
    eps = 1e-10
    my_lambda = 0

    while my_u > my_l:
        my_u = 0.95 - my_lambda
        my_l = 0.455 + 0.1 * my_lambda

        print("--------Thred_U------Thred_L--------------")
        print(my_u, my_l)

        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.size())
            # print(target.size())
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            # net.train()
            a, b = net(data)

            # for name, param in net.named_parameters():
            #     if param.requires_grad:
            #         print (name)
            # print(net.parameters())

            # norm = b.norm(p=2, dim=1, keepdim=True)
            # bn = b.div(norm.expand_as(b))
            # # bn = l2_norm(b)
            #
            # # print("----------")
            # # print(bn)
            # # print("----------")
            #
            # sim = torch.mm(bn, bn.t())
            #
            #
            #
            # pos_loc = torch.gt(sim, my_u).type(torch.float32)
            # neg_loc = torch.lt(sim, my_l).type(torch.float32)
            #
            # pos_entropy = torch.mul(-torch.log(torch.clamp(sim, eps, 1.0)), pos_loc)
            # neg_entropy = torch.mul(-torch.log(torch.clamp((1 - sim), eps, 1.0)), neg_loc)
            # loss_sum = torch.mean(pos_entropy + neg_entropy)
            # lf = LossFunc(my_u,my_l,eps)
            lf = LossFunc2()
            loss_sum = lf(b)

            if batch_idx%20==0:
                print(loss_sum.item())
                # print(sim)

            optimizer_model.zero_grad()
            loss_sum.backward()
            optimizer_model.step()

        my_lambda += 1.1 * 0.009

        print("------------my_lambda-----------------")
        print(my_lambda)

    # model.eval()
    # correct, total = 0, 0
        all_predictions, image_label = [], []
        all_predictions, image_label = np.array(all_predictions), np.array(image_label)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):

                if use_gpu:
                    data, labels = data.cuda(), labels.cuda()

                _, outputs = net(data)

                if use_gpu:
                    pred_cluster = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int)
                    data_labels = labels.data.cpu().numpy()
                else:
                    pred_cluster = torch.argmax(outputs, dim=1).numpy().astype(np.int)
                    data_labels = labels.data.numpy()

                acc = clustering_acc(data_labels, pred_cluster)
                nmi = NMI(data_labels, pred_cluster)
                ari = ARI(data_labels, pred_cluster)

                if (batch_idx + 1) % 20 == 0:
                    print(
                        "Batch {}/{}\t NMI {:.6f} | ARI : {:.6f} | ACC: {:.6f}".format(batch_idx + 1, len(test_loader), nmi,
                                                                                       ari, acc))

                all_predictions = np.concatenate((all_predictions, pred_cluster)).astype(np.int)
                image_label = np.concatenate((image_label, data_labels)).astype(np.int)

            acc = clustering_acc(image_label, all_predictions)
            nmi = NMI(image_label, all_predictions)
            ari = ARI(image_label, all_predictions)

            print('The finall testing NMI, ARI, ACC , my_acc are %f, %f, %f.' % (nmi, ari, acc))

        print()

    # loc =