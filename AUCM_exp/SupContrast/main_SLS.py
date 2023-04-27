from __future__ import print_function

import os
import sys
import argparse
import time
import math
from time import strftime, localtime


# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

# from main_ce import set_loader
from utils.util import AverageMeter, log_results, save_results
from utils.util import adjust_learning_rate, warmup_learning_rate
from utils.metrics import accuracy, get_all_metrics
from utils.util import set_optimizer, save_model
# from networks.resnet_big import SupConResNet, LinearClassifier
from networks.main import SupConResNet, LinearClassifier
from sklearn.metrics import roc_auc_score
from calibration_library.metrics import ECELoss, SCELoss
from dataset.datasets import set_loader
from solvers.losses import SupConLoss
import wandb

import json
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.8,
                        help='learning rate')
    parser.add_argument('--optim', type=str, default="sgd", help='optimizer')
    parser.add_argument('--lr_decay_epochs', type=str, default='500,700,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cls_type', type=str, default='binary', choices=['binary', 'multi'],
                        help='classification type: binary or multi-class')

    # model dataset
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'supcon','mdca','focal'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','c2','stl10','melanoma','imagenet_lt'], help='dataset')
    parser.add_argument('--imratio', type=float, default=0.1,
                        help='imbalance ratio')
    parser.add_argument('--gamma', type=float, default=1000, help='gamma for focal loss and AUCM loss')
    parser.add_argument('--alpha', type=float, default=0.75, help='alpha for focal loss')
    parser.add_argument('--margin', type=float, default=1.0, help='margin for AUCM loss')
    parser.add_argument('--shift_freq', type=int, default=5,
                        help='shift frequency')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--delta', type=float, default=0, help='delta for corruptions. Vary from 0 to 1')
    parser.add_argument('--prefix', type=str, default="testing", help='Extra string to be added for tracking purposes')
    parser.add_argument('--save_dir', type=str, default="~/temp_save_dir", help='Path where models will be saved')
    parser.add_argument('--freeze', type=int, default=0, help="Whether or not to freeze the encoder (1/0)")

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    print("ckpt is there")
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--seed', type=int, default=123, help='seed for random number')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '../../data/'
    opt.model_path = os.path.join(opt.save_dir,'{}_models'.format(opt.dataset))
    # opt.tb_path = './main_save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_name = 'SupSLSMDCA_{}_{}_im_{}_lr_{}_decay_{}_bsz_{}'.\
    opt.model_name = 'SupSLS_{}_{}_{}_{}_im_{}_lr_{}_decay_{}_bsz_{}_d_{}_temp_{}_trial_{}_epochs_{}_optim_{}_prefix_{}_freeze_{}'.\
        format(strftime("%d-%b", localtime()),opt.method, opt.dataset, opt.model,opt.imratio, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.delta, opt.temp, opt.trial, opt.epochs, opt.optim, opt.prefix, opt.freeze)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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
    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # if opt.dataset == 'cifar10':
    opt.n_cls = get_num_classes(opt)
    # elif opt.dataset == 'cifar100':
    #     opt.n_cls = 2
    # elif opt.dataset == 'c2':
    #     opt.n_cls = 2
    # else:
    #     raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt
from PIL import Image
from torch.utils.data import Dataset
from libauc.utils import ImbalancedDataGenerator #BUG:  this is using import from conda install
import numpy as np
from libauc.datasets import CAT_VS_DOG, CIFAR10, CIFAR100
from main_supcon import train as train_supcon

from solvers.losses import ClassficationAndMDCA, FocalLoss

from utils.deterministic import seed_everything
# SEED=123
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# opt = parse_option()
# seed_everything(opt.seed)
# from solvers.losses import loss_dict
# from loss import LogitNormLoss
# from loss import ClassficationAndMDCA
def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion1 = SupConLoss(temperature=opt.temp)
    if opt.loss == 'ce':
        criterion2 = torch.nn.CrossEntropyLoss()
    elif opt.loss == 'focal':
        criterion2 = FocalLoss(gamma=opt.gamma,alpha=opt.alpha)
    elif opt.loss == 'mdca':
        criterion2 = ClassficationAndMDCA(beta=opt.beta)
    else:
        criterion2 = torch.nn.CrossEntropyLoss()
    # criterion2 = loss_dict['LogitNorm']()
    # criterion2 = ClassficationAndMDCA(beta=0.1)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            # print(state_dict)
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion1, criterion2


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    classifier.train()
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    pred_total = []
    label_total = []

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print(labels)
        if labels.dim() == 2:
            labels = labels.squeeze(1) # Assert: Shape of labels is reduced to single dimension
        labels=labels.long()

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # freeze the weights of encoder
        if opt.freeze == 1:
            with torch.no_grad():
                features = model.encoder(images)
        else:
            features = model.encoder(images)

        output = classifier(features)
        loss = criterion(output, labels.long())
        # print(output)
        output = torch.nn.functional.softmax(output, dim=1)
        # print(output)
        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1,1))
        top1.update(acc1[0], bsz)
        # print(output)
        # print(output.shape)
        # if output.dim() == 2:
        #     print("reducing dimensions as its one dimensional")
        #     output=output[:, 1]

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: Not optimal in memory requirement setting
        pred_total.append(output.detach().cpu().numpy())
        label_total.append(labels.detach().cpu().numpy())
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
    print(pred_total.shape)
    print(label_total.shape)
    name='train'
    results = get_all_metrics('train', pred_total, label_total,opt)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))
    return results
    # auc = roc_auc_score(label_total, pred_total)

    # # convert probability of class 1 to 2d vector with probability of class 0 and 1
    # pred_total = np.stack([1 - pred_total, pred_total], axis=1)
    # eces = ECELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    # sces = SCELoss().loss(pred_total, label_total, n_bins=15, logits=False)

    # return losses.avg, auc, eces, sces

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

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
            output = classifier(model.encoder(images))
            loss = criterion(output, labels.long())
            output = torch.nn.functional.softmax(output, dim=1)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # add to total
            # print(output.shape)
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
    # compute AUC
    # print(len(pred_total), len(label_total))
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    print(pred_total.shape)
    print(label_total.shape)
    name='val'
    results = get_all_metrics('val', pred_total, label_total,opt)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))
    return results

    # auc = roc_auc_score(label_total, pred_total)
    # # convert probability of class 1 to 2d vector with probability of class 0 and 1
    # pred_total = np.stack([1 - pred_total, pred_total], axis=1)
    # eces = ECELoss().loss(pred_total, label_total, n_bins=15, logits=False)
    # sces = SCELoss().loss(pred_total, label_total, n_bins=15, logits=False)

    # # print stats
    # print(' * Acc@1 {top1.avg:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f}'
    #         .format(top1=top1, auc=auc, ece=eces, sce=sces))
    # return losses.avg, auc, eces, sces


def main(opt):
    best_results ={}

    wandb.init(
        # set the wandb project where this run will be logged
        entity="neelabh-madan",
        project="Refinement_exps",
        name=opt.model_name ,  
    )
    
    # track hyperparameters and run metadata
    wandb.config.update(opt)
    print(opt)

    # build data loader
    lossname=opt.loss
    opt.loss='supcon'
    train_loader1, val_loader = set_loader(opt)
    opt.loss=lossname
    train_loader2, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion1,criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if (epoch//opt.shift_freq)%2==0: # linear mode
            results = train(train_loader2, model, classifier, criterion2, optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, AUC:{:.2f}'.format( epoch, time2 - time1, results['train_auc']))

            # log_results(logger, results, epoch)
            log_results( results, epoch)

            # eval for one epoch
            results = validate(val_loader, model, classifier, criterion2, opt)
            # log_results(logger, results, epoch)
            log_results(results, epoch)

            if not best_results or results['val_auc'] > best_results['val_auc']:
                best_results = results
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
                

            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

        else:
            loss = train_supcon(train_loader1, model, criterion1, optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            # logger.log_value('train_loss', loss, epoch)
            wandb.log({'train_loss': loss,"epoch": epoch})
            
    print(best_results)
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce']))
    
    save_results(opt, best_results)


    return best_results

# def grid_search():
#     opt = parse_option()
#     if opt.loss ==
from utils.argparser import get_num_classes, update_option
from utils.util import save_results


def grid_search_focal(opt,train):
    gamma_list = [1,2,5]
    alpha_list = [0.25, 0.5, 0.75]
    best_results = {}
    for gamma in gamma_list:
        for alpha in alpha_list:
            opt.gamma = gamma
            opt.alpha = alpha
            opt = update_option(opt)
            results = train(opt)
            if not best_results or results['val_auc'] > best_results['val_auc']:
                best_results = results
                best_results['gamma'] = gamma
                best_results['alpha'] = alpha
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t alpha: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['alpha']))
    print(best_results)

    save_results(opt,best_results,name='_grid') 
    

def grid_search_mdca(opt,train):
    beta_list = [1,5, 10]
    best_results = {}
    for beta in beta_list:
        opt.beta = beta
        opt = update_option(opt)
        results = train(opt)
        if not best_results or results['val_auc'] > best_results['val_auc']:
            best_results = results
            best_results['beta'] = beta
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t alpha: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['alpha']))
    print(best_results)

    save_results(opt,best_results,name='_grid') 
    

def grid_search():
    opt = parse_option()
    seed_everything(opt.seed)
    if opt.loss == 'ce':
        main(opt)
    elif opt.loss == 'focal':
        grid_search_focal(opt,main)
    elif opt.loss == 'mdca':
        grid_search_mdca(opt,main)

if __name__ == '__main__':
    
    grid_search()
    # main(opt)
