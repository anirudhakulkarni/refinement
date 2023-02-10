# generate bash script for training and testing

losses=[ 'main_nll.py'
        # ,'main_aucm.py'
        ]
datasets=['cifar10','cifar100','c2','stl10']
models=['resnet20','densenet121']
imratios=[0.01,0.1]
gpus=[0,1,2,3,4,5,6,7]

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
        for loss in losses:
            for model in models:
                gpu=gpus[i%len(gpus)]
                i+=1
                script+='export CUDA_VISIBLE_DEVICES='+str(gpu)+'\n'
                script+='python3 '+loss
                if dataset=='cifar10' or dataset=='cifar100':
                    script+=' --batch_size 256 \ \n'
                else:
                    script+=' --batch_size 128 \ \n'
                script+='  --learning_rate 0.8 \ \n'
                script+='  --cosine \ \n'
                script+='  --imratio '+str(imratio)+' \ \n'
                script+='  --dataset '+dataset+' \ \n'
                script+='  --model '+model+' &\n'                
                script+='\n'
with open('nll.sh','w') as f:
    f.write(script)
