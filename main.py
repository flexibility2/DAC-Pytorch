import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets
# import models
import models3 as models
from utils import AverageMeter, Logger,ARI,NMI,clustering_acc
import warnings
warnings.filterwarnings('ignore')

# from center_loss import CenterLoss

parser = argparse.ArgumentParser("DAC Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    test_batch_size = 512
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size,test_batch_size=test_batch_size, use_gpu=use_gpu,
        num_workers=args.workers,
    )
    trainloader, testloader = dataset.trainloader, dataset.testloader

    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=dataset.num_classes)
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer_model = torch.optim.RMSprop(model.parameters(),lr=0.001)

    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        my_train(model,optimizer_model,trainloader,use_gpu=use_gpu)                         ##  train the model
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            my_test(model,trainloader,use_gpu=use_gpu)                                      ##   test the model

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

###################================================================================#####################################

class LossFunc(nn.Module):
    def __init__(self,my_u,my_l,eps):
        super(LossFunc,self).__init__()
        self.my_u = my_u
        self.my_l = my_l
        self.eps = eps

    def forward(self, label_feat):
        label_feat_norm = self.l2_norm(label_feat)
        sim_mat = torch.mm(label_feat_norm, label_feat_norm.t())

        pos_loc = torch.gt(sim_mat, self.my_u).type(torch.float)
        neg_loc = torch.lt(sim_mat, self.my_l).type(torch.float)

        pos_entropy = torch.mul(-torch.log(torch.clamp(sim_mat, self.eps, 1.0)), pos_loc)
        neg_entropy = torch.mul(-torch.log(torch.clamp((1 - sim_mat), self.eps, 1.0)), neg_loc)

        loss_sum = torch.mean(pos_entropy) + torch.mean(neg_entropy)

        return loss_sum

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output



def my_train(model,optimizer_model,trainloader,use_gpu):

    model.train()
    eps = 1e-10
    my_lambda = 0
    my_u = 0.95
    my_l = 0.455


    while my_u > my_l:

        my_u = 0.95 - my_lambda
        my_l = 0.455 + 0.1 * my_lambda

        my_loss = AverageMeter()

        print(my_u, my_l)

        for batch_idx, (data, labels) in enumerate(trainloader):    ## all the train data for one epoch

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            _,label_feat = model(data)

            # label_feat_norm = F.normalize(label_feat,p=2,dim=1)
            my_loss_func = LossFunc(my_u,my_l,eps)
            loss_sum = my_loss_func(label_feat)

            optimizer_model.zero_grad()
            loss_sum.backward()
            optimizer_model.step()

            my_loss.update(loss_sum.item(), label_feat.size(0))
            if (batch_idx + 1) % args.print_freq == 0:
                print("Batch {}/{}\t cur_loss {:.6f} |avg_loss {:.6f} ({:.6f})" \
                      .format(batch_idx + 1, len(trainloader), loss_sum.item(),my_loss.val, my_loss.avg))

        my_lambda += 1.1 * 0.009
        print(my_lambda)


def my_test(model,trainloader,use_gpu):

    model.eval()
    # correct, total = 0, 0
    all_predictions, image_label = [],[]
    all_predictions, image_label = np.array(all_predictions), np.array(image_label)


    with torch.no_grad():
        for batch_idx,(data, labels) in enumerate(trainloader):

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            _, outputs = model(data)

            if use_gpu:
                pred_cluster = torch.argmax(outputs,dim=1).cpu().numpy().astype(np.int)
                data_labels = labels.data.cpu().numpy()
            else:
                pred_cluster = torch.argmax(outputs,dim=1).numpy().astype(np.int)
                data_labels = labels.data.numpy()

            acc = clustering_acc(data_labels,pred_cluster)
            nmi = NMI(data_labels, pred_cluster)
            ari = ARI(data_labels, pred_cluster)

            if (batch_idx + 1) % args.print_freq == 0:
                print("Batch {}/{}\t NMI {:.6f} | ARI : {:.6f} | ACC: {:.6f}".format(batch_idx + 1, len(trainloader),nmi,ari,acc))

            all_predictions = np.concatenate((all_predictions,pred_cluster)).astype(np.int)
            image_label = np.concatenate((image_label,data_labels)).astype(np.int)

        acc = clustering_acc(image_label, all_predictions)
        nmi = NMI(image_label, all_predictions)
        ari = ARI(image_label, all_predictions)


        print('The finall testing NMI, ARI, ACC , my_acc are %f, %f, %f.' % (nmi, ari,acc))



if __name__ == '__main__':

    main()