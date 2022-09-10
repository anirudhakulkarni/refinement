# iterate on random seeds and check if the PACS is working

import os
import json
base=" python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --seed "
losses=["cross_entropy"]
import sys
args=sys.argv
# main function
for seed in range(int(args[1]),int(args[2])):
    for loss in losses:
        sces=[]
        # for a seed execute command and store the console output
        os.system(base+str(seed)+" --loss "+loss+" --target_type=art"+" 2> pacscheck/"+str(seed)+".txt")
        with open('train_results_anirudha.json') as f :
            data = json.load(f)
            sces.append(data[-1]['SCE'])
        # collect result from json object
        os.system(base+str(seed)+" --loss "+loss+" --target_type=cartoon"+" 2> pacscheck/"+str(seed)+".txt")
        with open('train_results_anirudha.json') as f :
            data = json.load(f)
            sces.append(data[-1]['SCE'])
        os.system(base+str(seed)+" --loss "+loss+" --target_type=sketch"+" 2> pacscheck/"+str(seed)+".txt")
        with open('train_results_anirudha.json') as f :
            data = json.load(f)
            sces.append(data[-1]['SCE'])
        # store the results in csv
        with open('pacscheck.csv', 'a') as f:
            f.write(str(seed)+","+str(sces[0])+","+str(sces[1])+","+str(sces[2])+","+str((sces[0]+sces[1]+sces[2])/3)+"\n")







