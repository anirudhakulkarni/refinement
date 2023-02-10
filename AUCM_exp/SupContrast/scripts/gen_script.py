# generate bash script for training and testing

files=[ 'main_nll.py'
        # ,'main_aucm.py'
        ]
datasets=['cifar10','cifar100','c2','stl10']
models=['resnet20','densenet121']
imratios=[0.01,0.1]
gpus=[0,1,2,3,4,5,6,7]
losses = ['focal']
gammas = [1,
          2,5
          ]
script='#!/bin/bash\n'
'''
sample:
python main_aucm.py --batch_size 256 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar10 \
  --model resnet20 &
'''
i=0
for imratio in imratios:
    for dataset in datasets:
        for file in files:
            for model in models:
                for gamma in gammas:
                    gpu=gpus[i%len(gpus)]
                    i+=1
                    script+='export CUDA_VISIBLE_DEVICES='+str(gpu)+'\n'
                    script+='python3 '+file
                    if dataset=='cifar10' or dataset=='cifar100':
                        script+=' --batch_size 512 \\\n'
                    else:
                        script+=' --batch_size 512 \\\n'
                    script+='  --loss '+losses[0]+' \\\n'
                    script+='  --gamma '+str(gamma)+' \\\n'
                    script+='  --learning_rate 0.8 \\\n'
                    script+='  --cosine \\\n'
                    script+='  --imratio '+str(imratio)+' \\\n'
                    script+='  --dataset '+dataset+' \\\n'
                    script+='  --model '+model+' &\n'                
                    script+='\n'
with open('nll.sh','w') as f:
    f.write(script)
