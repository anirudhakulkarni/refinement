# AUCM
# running on vision01
export CUDA_VISIBLE_DEVICES='2' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model resnet18 --imratio 0.1 --batch_size 127 --learning_rate 0.1"


export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model resnet18 --imratio 0.1 --batch_size 127 --learning_rate 0.1"


export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset stl10 --model resnet18 --imratio 0.1 --batch_size 32 --learning_rate 0.1"


export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset c2 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.1"


# CE
# running on cse01

export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar10 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar100 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset stl10 --model resnet18 --imratio 0.1 --batch_size 32 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset c2 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine"

# Focal
# running on cse01
export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar10 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar100 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset stl10 --model resnet18 --imratio 0.1 --batch_size 32 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset c2 --model resnet18 --imratio 0.1 --batch_size 128 --learning_rate 0.8"


# SupCon
# running on vision01

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset cifar10  --model resnet18  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset c2   --model resnet18 --ckpt save/SupCon/c2_models/SupCon_c2_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset cifar100   --model resnet18 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 32   --learning_rate 5   --imratio 0.1   --dataset stl10   --model resnet18 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth
"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
""

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
""
