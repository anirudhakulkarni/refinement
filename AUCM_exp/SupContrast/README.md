# How to run
python3 main_{lossType}.py

Example:
```
main_aucm.py
main_linear.py
main_linear_unfrz.py
main_nll_multi.py
main_nll.py
main_SLS.py
main_supcon.py
```

```
usage: argument for training [-h] [--print_freq PRINT_FREQ] [--save_freq SAVE_FREQ] [--batch_size BATCH_SIZE]
                             [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                             [--learning_rate2 LEARNING_RATE2] [--lr_decay_epochs LR_DECAY_EPOCHS]
                             [--lr_decay_rate LR_DECAY_RATE] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                             [--loss {ce,supcon,focal,aucm,aucs,ce_linear,ifl,dca,brier,logitnorm}] [--model MODEL]
                             [--dataset DATASET] [--imratio IMRATIO] [--size SIZE] [--margin MARGIN] [--gamma GAMMA]
                             [--alpha ALPHA] [--delta DELTA] [--temp TEMP] [--shift_freq SHIFT_FREQ] [--no_grid]
                             [--cosine] [--syncBN] [--warm] [--trial TRIAL] [--seed SEED] [--cls_type {binary,multi}]
                             [--stages STAGES] [--ckpt CKPT]

options:
  -h, --help            show this help message and exit
  --print_freq PRINT_FREQ
                        print frequency
  --save_freq SAVE_FREQ
                        save frequency
  --batch_size BATCH_SIZE
                        batch_size
  --num_workers NUM_WORKERS
                        num of workers to use
  --epochs EPOCHS       number of training epochs
  --learning_rate LEARNING_RATE
                        learning rate
  --learning_rate2 LEARNING_RATE2
                        learning rate
  --lr_decay_epochs LR_DECAY_EPOCHS
                        where to decay lr, can be a list
  --lr_decay_rate LR_DECAY_RATE
                        decay rate for learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --momentum MOMENTUM   momentum
  --loss {ce,supcon,focal,aucm,aucs,ce_linear,ifl,dca,brier,logitnorm}
  --model MODEL
  --dataset DATASET     dataset
  --imratio IMRATIO     imbalance ratio for binary classification, Imbalance factor for Long tail dataset
  --size SIZE           parameter for RandomResizedCrop
  --margin MARGIN       margin for AUCM loss
  --gamma GAMMA         gamma for focal loss and AUCM loss
  --alpha ALPHA         alpha for focal loss
  --delta DELTA         delta for corruptions. Vary from 0 to 1
  --temp TEMP           temperature
  --shift_freq SHIFT_FREQ
                        shift frequency
  --no_grid             no hyperparameter tunning
  --cosine              using cosine annealing
  --syncBN              using synchronized batch normalization
  --warm                warm-up for large batch training
  --trial TRIAL         id for recording multiple runs
  --seed SEED           seed for random number
  --cls_type {binary,multi}
                        classification type: binary or multi-class
  --stages STAGES       2 stage training. parse initial large epochs and then frequency of small epochs
  --ckpt CKPT           path to pre-trained model
```


# Useful scripts:

main.sh has many examples on how to run particular example





# TODO
1. ~~ECE and SCE for multiclass~~
2. ~~Batch normalization~~ 
3. ~~Imagenet LT~~
4. ~~Cifar100 LT~~
5. ~~Cifar10 LT~~
6. Naturalist LT
7. Shared data directory
8. aucm with augmentation
9. training procedure as an algo
10. training procedure in paper
11. ~~adam vs sgd result difference~~ Use SGD without momentum. Momentum harmed
12. tsne plot
13. find the error example





repeatedly used commands
Visualization:
https://github.com/Jonathan-Pearce/calibration_library/blob/master/visualization.py

# kill all screen sessions that start with "52*"
screen -ls | grep 52 | cut -d. -f1 | awk '{print $1}' | xargs kill~~
