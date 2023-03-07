# AUCM
# running on vision01
export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


# CE
# running on cse01

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450"

# Focal
# running on cse01
export CUDA_VISIBLE_DEVICES='1,2'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='3,4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='5,6'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.8 --epochs 500 --lr_decay_epochs 250,350,450"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --epochs 500 --lr_decay_epochs 250,350,450"


# SupCon
# already done
export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar10 --imratio 0.01 --model resnet18 --cosine --epochs 450"

export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --imratio 0.01 --model resnet18 --cosine --epochs 450"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset stl10 --imratio 0.01 --model resnet18 --cosine --epochs 450"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.01 --model resnet18 --cosine --epochs 450"

# # SLS
# # running on vision01

export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c  \
"python main_SLS.py --batch_size 512 --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset cifar10  --model resnet18  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512  --epochs 50 --learning_rate 0.1   --imratio 0.01   --dataset c2   --model resnet18 --ckpt save/SupCon/c2_models/SupCon_c2_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset cifar100   --model resnet18 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset stl10   --model resnet18 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --loss mdca --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset cifar10  --model resnet18  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --loss mdca  --epochs 50 --learning_rate 0.1   --imratio 0.01   --dataset c2   --model resnet18 --ckpt save/SupCon/c2_models/SupCon_c2_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --loss mdca --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset cifar100   --model resnet18 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth && \
python main_SLS.py --batch_size 512 --loss mdca --epochs 50  --learning_rate 0.1   --imratio 0.01   --dataset stl10   --model resnet18 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth


"

# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# ""

# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# ""


export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset c2  --model resnet18  --epochs 50 --loss ce --ckpt main_save/SupCon/c2_models/SupCon_c2_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth "
export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset c2  --model resnet18  --epochs 50 --loss mdca --ckpt main_save/SupCon/c2_models/SupCon_c2_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth "
export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset stl10  --model resnet18  --epochs 50 --loss mdca --ckpt ./main_save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth" 
export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset stl10  --model resnet18  --epochs 50 --loss ce --ckpt ./main_save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth "

# remaining
export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset cifar100  --model resnet18  --epochs 50 --loss mdca --ckpt ./main_save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth" 
export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset cifar100  --model resnet18  --epochs 50 --loss ce --ckpt ./main_save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth" 
export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset cifar10  --model resnet18  --epochs 50 --loss mdca --ckpt ./main_save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth" 
export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.01   --dataset cifar10  --model resnet18  --epochs 50 --loss ce --ckpt ./main_save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.01_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth" 



# IFL
export CUDA_VISIBLE_DEVICES='6' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss ifl --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss ifl --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss ifl --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss ifl --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


# BrierScore
export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss brier --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss brier --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss brier --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss brier --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"



# DCA
export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss dca --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss dca --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss dca --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss dca --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"

# LogitNorm
export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss logitnorm --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss logitnorm --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss logitnorm --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"


export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss logitnorm --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --epochs 500 --lr_decay_epochs 250,350,450"



# Corrupted
export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450 --delta 10.0"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss aucm --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine --epochs 500 --lr_decay_epochs 250,350,450 --delta 10.0 --no_grid --gamma 500 --margin 0.7"

export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.01 --model resnet18 --cosine --epochs 1000"
