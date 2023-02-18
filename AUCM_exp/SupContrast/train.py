from __future__ import print_function
import json


import os
import sys
from utils.argparser import parse_option
from utils.deterministic import seed_everything
from utils.util import save_model, adjust_learning_rate, log_results, save_results
import time
opt = parse_option()
seed_everything(opt.seed)





import tensorboard_logger as tb_logger
import torch

from solvers.model import set_model
from dataset.datasets import set_loader
from solvers.optimizer import set_optimizer
from solvers.runners import get_train_test
from solvers.grid_search import grid_search

def train(opt):
    best_results = {}
    if 'sls' in opt.loss:
        lossname=opt.loss
        opt.loss='supcon'
        train_loader2, val_loader = set_loader(opt)
        opt.loss='ce'
        train_loader1, val_loader = set_loader(opt)
        opt.loss=lossname
    else:
        train_loader1, val_loader = set_loader(opt)
    model, criterion, criterion2 = set_model(opt)
    optimizer1, optimizer2 = set_optimizer(opt, model, criterion)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    train_one_epoch, test_one_epoch = get_train_test(opt) 

    for epoch in range(1, opt.epochs + 1):
        try:
            time1 = time.time()
            if 'sls' in opt.loss:
                results = train_one_epoch((train_loader1, train_loader2), model, (criterion, criterion2), (optimizer1,optimizer2), epoch, opt)
            else:
                print(criterion,optimizer1)
                results = train_one_epoch(train_loader1, model, criterion, optimizer1, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            if opt.loss != 'sls':
                log_results(logger, results, epoch)

            results = test_one_epoch(val_loader, model, criterion, opt, val=True)
            log_results(logger, results, epoch)

            if not best_results or results['val_auc'] > best_results['val_auc']:
                best_results = results
                save_file = os.path.join(opt.save_folder, 'best.pth')
                save_model(model, optimizer1, opt, epoch, save_file)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            inp_exit = input('Exit? [y/n]')
            save_file = os.path.join(opt.save_folder, 'last.pth')
            save_model(model, optimizer1, opt, epoch, save_file)
            if inp_exit == 'y':
                break
            else:
                continue
                
    best_results['margin']=opt.margin                                                                                 
    best_results['gamma']=opt.gamma 

    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce']))
    
    save_results(opt, best_results)
    return best_results
        


train(opt)
# grid_search(opt,train)