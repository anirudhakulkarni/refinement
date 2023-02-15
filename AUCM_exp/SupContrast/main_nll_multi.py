from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.main import SupCEResNet, SupAUCMResNet
from utils.metrics import auc_m
from calibration_library.metrics import ECELoss, SCELoss
from dataset.datasets_multiclass import set_loader
import json
from loss import FocalLoss
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--imratio', type=float, default=0.1,
                        help='imbalance ratio')
    parser.add_argument('--gamma', type=float, default=1, help='gamma in focal loss')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '../../data/'
    if opt.dataset == 'imagenet':
        opt.data_folder = '../../data/imagenet/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.loss == 'ce':
        opt.model_name = 'MultiCE_{}_{}_im_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
            format(opt.dataset, opt.model, opt.imratio, opt.learning_rate, opt.weight_decay,
                opt.batch_size, opt.trial)
    else:
        opt.model_name = 'MultiFL_{}_{}_im_{}_lr_{}_decay_{}_bsz_{}_gamma_{}_trial_{}'.\
            format(opt.dataset, opt.model, opt.imratio, opt.learning_rate, opt.weight_decay,
                   opt.batch_size, opt.gamma, opt.trial)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    traditional_datas=['balance', 'dermatology', 'ecoli', 'new-thyroid', 'pageblocks', 'segmentImb', 'shuttle', 'svmguide2', 'yeast']
    traditional_n_cls = [3,6,8,3,5,7,5,3,10]

    if opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset in traditional_datas:
        opt.n_cls = traditional_n_cls[traditional_datas.index(opt.dataset)]
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt
from PIL import Image
from torch.utils.data import Dataset
from libauc.utils import ImbalancedDataGenerator #BUG:  this is using import from conda install
import numpy as np
from libauc.datasets import CAT_VS_DOG, CIFAR10, CIFAR100
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
SEED=123
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    if opt.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif opt.loss == 'focal':
        criterion = FocalLoss(gamma=opt.gamma)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    pred_total = []
    label_total = []

    for idx, (images, labels) in enumerate(train_loader):
        # print(labels)
        # if labels have 2d shape, squeeze it
        if labels.dim() == 2:
            labels = labels.squeeze(1) # Assert: Shape of labels is reduced to single dimension
        labels=labels.long()
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        logits = model(images)
        loss = criterion(logits, labels)
        output = torch.softmax(logits,dim=1)
        
        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # NOTE: use the output of maximum class is to be given to ECE loss
        # if output.dim() == 2:
        #     output=output[:, 1]
        # output = output.contiguous().view(-1, 1)
        # print(output.shape)
        pred_total.append(output.detach().cpu().numpy())
        label_total.append(labels.detach().cpu().numpy())

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    auc = auc_m(label_total, pred_total)

    # convert probability of class 1 to 2d vector with probability of class 0 and 1
    pred_total = np.stack([1 - pred_total, pred_total], axis=1)
    # TODO: SOlve this for multi-class
    # eces = ECELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    # sces = SCELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    eces, sces = 0, 0

    return losses.avg, auc, eces, sces
    # return losses.avg, auc



def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pred_total = []
        label_total = []
        for idx, (images, labels) in enumerate(val_loader):
            labels=labels.long()

            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            if labels.dim() == 2:
                labels = labels.squeeze(1) # Assert: Shape of labels is reduced to single dimension

            # forward
            logits = model(images)
            loss = criterion(logits, labels)
            output = torch.softmax(logits, dim=1)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # add to total
            # print(output.shape) # ASSERT: output shape has 2 dimensions
            # use the probability of the positive class only
            # if output.dim() == 2:
            #     output=output[:, 1]
            # output = output.contiguous().view(-1, 1)
            # print(output.shape)
            pred_total.append(output.cpu().numpy())
            label_total.append(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    auc = auc_m(label_total, pred_total)
    # convert probability of class 1 to 2d vector with probability of class 0 and 1
    pred_total = np.stack([1 - pred_total, pred_total], axis=1)
    # TODO: SOlve this for multi-class
    # eces = ECELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    # sces = SCELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    eces, sces = 0, 0
    # print stats
    print(' * Acc@1 {top1.avg:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f}'
            .format(top1=top1, auc=auc, ece=eces, sce=sces))
    return losses.avg, auc, eces, sces
    # return losses.avg, auc


def main():
    best_auc, best_ece, best_sce = 0, 0, 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_auc,eces,sces = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_auc', train_auc, epoch)
        logger.log_value('train_ece', eces, epoch)
        logger.log_value('train_sce', sces, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_auc, eces, sces = validate(val_loader, model, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_auc', val_auc, epoch)
        logger.log_value('val_ece', eces, epoch)
        logger.log_value('val_sce', sces, epoch)
        if val_auc > best_auc:
            best_auc = val_auc
            best_ece = eces
            best_sce = sces
            save_file = os.path.join(opt.save_folder, 'best.pth')
            save_model(model, optimizer, opt, epoch, save_file)


    print('best AUC: {:.5f}\t best ECE: {:.5f}\t best SCE: {:.5f}'.format(best_auc, best_ece, best_sce))

    # save the results in a json file
    jsonfile='results.json'
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump({}, f)
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as f:
            results = json.load(f)
    results[opt.save_folder] = {
        'model': opt.model,
        'dataset': opt.dataset,
        'batch_size': opt.batch_size,
        'imratio': opt.imratio,
        'best_auc': best_auc, 'best_ece': best_ece, 'best_sce': best_sce,
                                
                                }
    with open(jsonfile, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()