'''
Sample Json:
{
    "./save/SupCon/c2_models/SupAUCM_c2_resnet20_im_0.1_lr_0.8_decay_0.0001_bsz_128_trial_0_cosine": {
        "best_auc": 0.9592023289273929,
        "best_ece": 0.012091518080234537,
        "best_sce": 0.043754516517370945
    },
}

Sample CSV:
Dataset, Model, ImRatio, AUC, ECE, SCE
'''

import json

with open('../results.json','r') as f:
    results=json.load(f)

import pandas as pd

# create a dataframe
df = pd.DataFrame(columns=['Dataset','Model','Loss', 'ImRatio','AUC','ECE','SCE'])
for result in results:
    dataset=result.split('_')[2]
    model=result.split('_')[3]
    imratio = result.split('_')[5]
    auc = results[result]['best_auc']
    ece = results[result]['best_ece']
    sce = results[result]['best_sce']
    loss = result.split('_')[1][10:]
    df = pd.concat([df,pd.DataFrame([[dataset,model,loss,imratio,auc,ece,sce]],columns=['Dataset','Model','Loss', 'ImRatio','AUC','ECE','SCE'])])
# sort
df = df.sort_values(by=['ImRatio','Model','Dataset'])
df.to_csv('results.csv',index=False)



'''
Create a table like this
Table 1: Testing AUC on benchmark datasets with imratio=1%.
Dataset CE Focal AUC-S AUC-M
C2 (D) 0.718±0.018 0.713±0.009 0.803±0.018 0.809±0.016
C10 (D) 0.698±0.017 0.700±0.007 0.745±0.010 0.760±0.006
S10 (D) 0.641±0.032 0.660±0.027 0.669±0.070 0.703±0.030
C100 (D) 0.588±0.011 0.591±0.017 0.607±0.010 0.614±0.016
C2 (R) 0.730±0.028 0.724±0.020 0.748±0.007 0.756±0.017
C10 (R) 0.690±0.011 0.681±0.011 0.702±0.015 0.715±0.008
S10 (R) 0.641±0.021 0.634±0.024 0.645±0.029 0.659±0.020
C100 (R) 0.563±0.015 0.565±0.022 0.587±0.017 0.596±0.016

'''

table = pd.DataFrame(columns=['Dataset','CE','Focal','AUC-S','AUC-M'])

for dataset in ['c2','cifar10','stl10','cifar100']:
    # choose only 0.01 imratio
    # c2, c10, s10, c100 are the datasets
    # D stands for densenet121 model
    # R stands for resnet20 model
    
    table = pd.concat([table,pd.DataFrame([[dataset+' (D)',
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='densenet121') & (df['Loss']=='' )]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='densenet121') & (df['Loss']=='focal')]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='densenet121') & (df['Loss']=='AUC-S')]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='densenet121') & (df['Loss']=='AUCM')]['AUC'].mean()]],columns=['Dataset','CE','Focal','AUC-S','AUC-M'])])
    
for dataset in ['c2','cifar10','stl10','cifar100']:
    table = pd.concat([table,pd.DataFrame([[dataset+' (R)',
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='resnet20') & (df['Loss']=='')]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='resnet20') & (df['Loss']=='focal')]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='resnet20') & (df['Loss']=='AUC-S')]['AUC'].mean(),
                                            df[(df['Dataset']==dataset) & (df['ImRatio']=='0.01') & (df['Model']=='resnet20') & (df['Loss']=='AUCM')]['AUC'].mean()]],columns=['Dataset','CE','Focal','AUC-S','AUC-M'])])
    
table.to_csv('table.csv',index=False)

