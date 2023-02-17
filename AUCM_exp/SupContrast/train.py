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
    train_loader, val_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model, criterion)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print(opt.tb_folder)
    train_one_epoch, test_one_epoch = get_train_test(opt) 
    for epoch in range(1, opt.epochs + 1):

        time1 = time.time()
        results = train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        log_results(logger, results, epoch)

        results = test_one_epoch(val_loader, model, criterion, opt, val=True)
        log_results(logger, results, epoch)

        if not best_results or results['val_auc'] > best_results['val_auc']:
            best_results = results
            save_file = os.path.join(opt.save_folder, 'best.pth')
            save_model(model, optimizer, opt, epoch, save_file)



    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce']))
    
    save_results(opt, best_results)
    return best_results
        


# train()
grid_search(opt,train)