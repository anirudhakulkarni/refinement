from __future__ import print_function
import json


import os
import sys
from .utils.argparser import parse_option
from .utils.deterministic import seed_everything
from .utils.util import save_model, adjust_learning_rate, log_results
import time
opt = parse_option()
seed_everything(opt.seed)





import tensorboard_logger as tb_logger
import torch

from solvers.model import set_model
from dataset.datasets import set_loader
from solvers.optimizer import set_optimizer
from .solvers.runners import train_epoch_AUCM, test_epoch_AUCM


def train():
    best_results = {}
    train_loader, val_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model, criterion)
    logger = tb_logger.Logger(logdir=opt.log_dir, flush_secs=2)
    
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        results = train_epoch_AUCM(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        log_results(logger, results, epoch)

        results = test_epoch_AUCM(val_loader, model, criterion, opt)
        log_results(logger, results, epoch)

        if not best_results or results['val_auc'] > best_results['val_auc']:
            best_results = results
            save_file = os.path.join(opt.save_folder, 'best.pth')
            save_model(model, optimizer, opt, epoch, save_file)



    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce']))
    
    # save the results in a json file
    jsonfile='results.json'
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump({}, f)
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as f:
            result_file = json.load(f)
            
    result_file[opt.save_folder] = {
        'model': opt.model,
        'dataset': opt.dataset,
        'batch_size': opt.batch_size,
        'margin': opt.margin,
        'imratio': opt.imratio,
        'best_results': best_results,
                                
                                }
    with open(jsonfile, 'w') as f:
        json.dump(result_file, f, indent=4)
        
    return best_results
        

def grid_search():
    gamma_list = [100,300,500,700,1000]
    margin_list = [0.1,0.3,0.5,0.7,1.0]
    best_results = {}    
    # iterate over the grid
    for gamma in gamma_list:
        for margin in margin_list:
            # set the parameters
            opt.gamma = gamma
            opt.margin = margin
            
            results = train()
            if not best_results or results['val_auc'] > best_results['val_auc']:
                best_results = results
                best_results['gamma'] = gamma
                best_results['margin'] = margin
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t margin: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['margin']))
    print(best_results)
    
    
    # save the results in a json file
    jsonfile='results.json'
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump({}, f)
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as f:
            result_file = json.load(f)
    
    result_file[opt.save_folder+'_grid'] = {
        'model': opt.model,
        'dataset': opt.dataset,
        'batch_size': opt.batch_size,
        'margin': opt.margin,
        'imratio': opt.imratio,
        'best_results': best_results,
    }
    
    with open(jsonfile, 'w') as f:
        json.dump(result_file, f, indent=4)
    
            