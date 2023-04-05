import os
import json
from utils.util import save_results
from utils.argparser import update_option
def is_no_grid(loss):
    is_no_grid_dict = ['brier','dca','logitnorm','ls']
    if loss in is_no_grid_dict:
        return True
    return False
def grid_search(opt,train):
    if opt.loss == 'aucm':
        grid_search_AUCM(opt,train)
    elif opt.loss == 'focal':
        grid_search_focal(opt,train)
    elif opt.loss == 'aucs':
        grid_search_AUCS(opt,train)
    elif opt.loss == 'ce' or opt.loss == 'ls':
        train(opt)
    elif opt.loss == 'ifl':
        grid_search_ifl(opt,train)
    elif is_no_grid(opt.loss):
        train(opt)
    else:
        raise ValueError('Unknown loss: {}'.format(opt.loss))
    
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

def grid_search_ifl(opt,train):
    gamma_list = [1,2,3]
    best_results = {}
    for gamma in gamma_list:
        opt.gamma = gamma
        opt = update_option(opt)
        results = train(opt)
        if not best_results or results['val_auc'] > best_results['val_auc']:
            best_results = results
            best_results['gamma'] = gamma
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t alpha: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['alpha']))
    print(best_results)

    save_results(opt,best_results,name='_grid') 
    
def grid_search_AUCM(opt,train):
    gamma_list = [100,300,500,700,1000]
    margin_list = [0.1,0.3,0.5,0.7,1.0]
    best_results = {}    
    # iterate over the grid
    for gamma in gamma_list:
        for margin in margin_list:
            # set the parameters
            opt.gamma = gamma
            opt.margin = margin
            opt = update_option(opt)
            
            results = train(opt)
            if not best_results or results['val_auc'] > best_results['val_auc']:
                best_results = results
                best_results['gamma'] = gamma
                best_results['margin'] = margin
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t margin: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['margin']))
    print(best_results)

    save_results(opt,best_results,name='_grid') 

def grid_search_AUCS(opt,train):
    gamma_list = [100,300,500,700,1000]
    best_results = {}    
    # iterate over the grid
    for gamma in gamma_list:
        # set the parameters
        opt.gamma = gamma
        opt = update_option(opt)
        
        results = train(opt)
        if not best_results or results['val_auc'] > best_results['val_auc']:
            best_results = results
            best_results['gamma'] = gamma
            best_results['margin'] = 1
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}\t gamma: {:.10f}\t margin: {:.10f}'.format(
        best_results['val_auc'], best_results['val_ece'], best_results['val_sce'], best_results['gamma'], best_results['margin']))
    print(best_results)

    save_results(opt,best_results,name='_grid') 
