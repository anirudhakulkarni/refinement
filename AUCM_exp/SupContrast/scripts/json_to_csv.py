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
df = df.sort_values(by=['Dataset','Model','ImRatio'])
df.to_csv('results.csv',index=False)

