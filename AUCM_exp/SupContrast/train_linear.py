from __future__ import print_function
from solvers.grid_search import grid_search
from solvers.runners import get_train_test, get_train_test_linear
from solvers.optimizer import set_optimizer
from dataset.datasets import set_loader
from solvers.model import set_model, set_model_linear
import torch
import tensorboard_logger as tb_logger
import json


import os
import sys
from utils.argparser import parse_option
from utils.deterministic import seed_everything
from utils.util import save_model, adjust_learning_rate, log_results, save_results
import time
opt = parse_option()
seed_everything(opt.seed)


def train(opt):
    best_results = {}
    lossname=opt.loss
    opt.loss='supcon'
    train_loader1, val_loader = set_loader(opt)
    opt.loss=lossname
    train_loader2, val_loader = set_loader(opt)

    model, classifier, criterion1,criterion2 = set_model_linear(opt)
    optimizer = set_optimizer(opt, model, criterion1)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    train_epoch_SupCon, train_epoch_Linear, test_one_epoch = get_train_test_linear(opt)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        try:
            time1 = time.time()
            if (epoch//opt.shift_freq)%2==0: # linear mode
                results = train_epoch_Linear(train_loader2, model, classifier, criterion2, optimizer, epoch, opt)
                time2 = time.time()
                print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
                log_results(logger, results, epoch)
                results = test_one_epoch(
                    val_loader, model, classifier, criterion2, opt, val=True)
                log_results(logger, results, epoch)

                if not best_results or results['val_auc'] > best_results['val_auc']:
                    best_results = results
                    save_file = os.path.join(opt.save_folder, 'best.pth')
                    save_model(model, optimizer, opt, epoch, save_file)
            else:
                # supcon mode
                results = train_epoch_SupCon(train_loader1, model, classifier, criterion1, optimizer, epoch, opt)
                time2 = time.time()
                print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            inp_exit = input('Exit? [y/n]')
            if inp_exit == 'y':
                break
            else:
                save_file = os.path.join(opt.save_folder, str(epoch)+'.pth')
                save_model(model, optimizer, opt, epoch, save_file)
                continue
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, epoch, save_file)

    best_results['margin'] = opt.margin
    best_results['gamma'] = opt.gamma

    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce']))

    save_results(opt, best_results)
    return best_results


train(opt)
# grid_search(opt,train)
