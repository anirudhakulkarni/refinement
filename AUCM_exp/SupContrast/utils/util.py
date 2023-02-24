from __future__ import print_function

import math
import time
import numpy as np
import torch
import torch.optim as optim
import os
import json
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving at {}...'.format(save_file))
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def log_results(logger, results, epoch):
    for key, value in results.items():
        logger.log_value(key, value, epoch)
        
    
def save_results(opt,best_results,name=''):  
    # save the results in a json file
    jsonfile='results.json'
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump({}, f)
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as f:
            result_file = json.load(f)
    print(best_results)
    if 'margin' not in best_results:
        best_results['margin']=-1
    if 'gamma' not in best_results:
        best_results['gamma']=-1
    if 'alpha' not in best_results:
        best_results['alpha']=-1
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    result_file[opt.save_folder+name] = {
        'time': current_time,
        'model': opt.model,
        'dataset': opt.dataset,
        'batch_size': opt.batch_size,
        'imratio': opt.imratio,
        'best_results': best_results,
    }
    
    with open(jsonfile, 'w') as f:
        json.dump(result_file, f, indent=4)
    
