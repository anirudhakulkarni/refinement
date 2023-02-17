import time
import torch
from utils.util import AverageMeter, warmup_learning_rate
from utils.metrics import get_all_metrics, accuracy
import numpy as np
import sys
def train_epoch_AUCM(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    pred_total = []
    label_total = []
    if epoch in opt.lr_decay_epochs:
        optimizer.update_regularizer(decay_factor=10) # decrease learning rate by 10x & update regularizer

    for idx, (images, labels) in enumerate(train_loader):
        if labels.dim() == 2:
            labels = labels.squeeze(1) 
            # Assert: Shape of labels is reduced to single dimension

        labels=labels.long()
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        logits = model(images)
        output = torch.sigmoid(logits)
        loss = criterion(output, labels)
        # print(output)
        # output = torch.nn.functional.normalize(output, p=2, dim=1)
        # print(output)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # remove one dimension. Number of classes = 1. Hence sigmoid works instead of softmax
        output = torch.squeeze(output)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    results = get_all_metrics('train', pred_total, label_total, opt)
    # print(results)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results['train_top1'], auc=results['train_auc'], ece=results['train_ece'], sce=results['train_sce']))

    return results

def test_epoch_AUCM(test_loader, model, criterion, opt, val=False):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pred_total = []
        label_total = []
        for idx, (images, labels) in enumerate(test_loader):
            labels=labels.long()
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            if labels.dim() == 2:
                labels = labels.squeeze(1) 
                # Assert: Shape of labels is reduced to single dimension

            # forward
            logits = model(images)
            output = torch.sigmoid(logits)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            # remove one dimension. Number of classes = 1. Hence sigmoid works instead of softmax
            output = torch.squeeze(output)

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
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    name = 'test' if not val else 'val'
    results = get_all_metrics(name, pred_total, label_total,opt)

    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))
    return results

def train_epoch_CE(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    pred_total = []
    label_total = []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_decay_epochs, gamma=opt.lr_decay_rate)
    for idx, (images, labels) in enumerate(train_loader):
        if labels.dim() == 2:
            labels = labels.squeeze(1) 
            # Assert: Shape of labels is reduced to single dimension

        labels=labels.long()
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        logits = model(images)
        loss = criterion(logits, labels)
        output = torch.softmax(logits,dim=1) #will not work for 1d
        # output = torch.sigmoid(logits)
        # print(output)
        # output = torch.nn.functional.normalize(output, p=2, dim=1)
        # print(output)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        # remove one dimension. Number of classes = 1. Hence sigmoid works instead of softmax
        # output = torch.squeeze(output)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    scheduler.step()
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    results = get_all_metrics('train', pred_total, label_total,opt)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results['train_top1'], auc=results['train_auc'], ece=results['train_ece'], sce=results['train_sce']))

    return results

def get_train_test(opt):
    if opt.loss == 'ce' or opt.loss == 'focal':
        train_epoch = train_epoch_CE
    elif opt.loss == 'aucm' or opt.loss == 'aucs':
        train_epoch = train_epoch_AUCM
    else:
        raise ValueError('Unknown loss function: {}'.format(opt.loss))
    return train_epoch, test_epoch_AUCM