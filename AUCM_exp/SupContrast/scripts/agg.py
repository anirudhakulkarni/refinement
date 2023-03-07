'''
    "./save/SupCon/c2_models/binary_aucm_c2_resnet18_im_0.01_lr_0.1_bsz_128_g_1000_m_1.0_epochs_100_grid": {
        "model": "resnet18",
        "dataset": "c2",
        "batch_size": 128,
        "imratio": 0.01,
        "best_results": {
            "val_ece": 0.15772634114027023,
            "val_mce": 0.21911499842558757,
            "val_oel": 0.13346147197932218,
            "val_sce": 0.16089490889385344,
            "val_ace": 0.15939663762953135,
            "val_top1": 63.47999954223633,
            "val_auc": 0.690642768727807,
            "margin": 1.0,
            "gamma": 300,
            "alpha": -1
        }
'''
def filter(jf,te):
    # this json object to filtered json object
    # filter out those which have "best_results" key
    result={}
    content = jf[te]
    if 'aucm' in te:
        result['loss'] = 'aucm'
    if 'ce' in te:
        result['loss'] = 'ce'
    if 'focal' in te:
        result['loss'] = 'focal'
    if 'sls' in te:
        result['loss'] = 'sls'
    result['model']=content['model']
    result['dataset']=content['dataset']
    result['imratio']=content['imratio']
    result['ece']=content['best_results']['val_ece']
    result['mce']=content['best_results']['val_mce']
    result['oel']=content['best_results']['val_oel']
    result['sce']=content['best_results']['val_sce']
    result['ace']=content['best_results']['val_ace']
    result['top1']=content['best_results']['val_top1']
    result['auc']=content['best_results']['val_auc']

    return result

def aggregator_function(jsonfile):
    # iterate on jsonfile and select those which satisfy following criteria
    # "binary_aucm_c2_resnet18_im_0.01_lr_0.1_bsz_128" in te
    results = []
    for te in jsonfile:
        if "resnet18" in te  and "focal" in te and "grid" in te:
            
            results.append(filter(jsonfile,te))
    return results


import json

with open('./results.json','r') as f:
    results=json.load(f)

import pandas as pd

df=pd.DataFrame(aggregator_function(results))
df.to_csv('results.csv',index=False)