# AUCM
export CUDA_VISIBLE_DEVICES='1' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model resnet50 --imratio 0.1 --batch_size 127 --learning_rate 0.1 && \
python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model resnet50 --imratio 0.1 --batch_size 127 --learning_rate 0.1"


export CUDA_VISIBLE_DEVICES='2' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset stl10 --model resnet50 --imratio 0.1 --batch_size 32 --learning_rate 0.1 && \
python3 train.py --cls_type binary --loss aucm --dataset c2 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.1"


# CE
export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar10 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine && \
python train.py --cls_type binary --loss ce --dataset cifar100 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine && \
python train.py --cls_type binary --loss ce --dataset stl10 --model resnet50 --imratio 0.1 --batch_size 32 --learning_rate 0.8 --cosine && \
python train.py --cls_type binary --loss ce --dataset c2 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8 --cosine"

# Focal
export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar10 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8 && \
python train.py --cls_type binary --loss focal --dataset cifar100 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8 && \
python train.py --cls_type binary --loss focal --dataset stl10 --model resnet50 --imratio 0.1 --batch_size 32 --learning_rate 0.8 && \
python train.py --cls_type binary --loss focal --dataset c2 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.8"


# SupCon
export CUDA_VISIBLE_DEVICES='7' 
screen -dm bash -c  \
"python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset cifar10 --imratio 0.1 --model resnet50 --cosine && \
python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --imratio 0.1 --model resnet50 --cosine && \
python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset stl10 --imratio 0.1 --model resnet50 --cosine && \
python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.1 --model resnet50 --cosine"



#TODO: fill in the paths correctly
# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# "python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset cifar10  --model resnet50  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
# python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset c2   --model resnet50 --ckpt save/SupCon/c2_models/SupCon_c2_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
# python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1   --dataset cifar100   --model resnet50 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
# python main_SLS.py --batch_size 32   --learning_rate 5   --imratio 0.1   --dataset stl10   --model resnet50 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth"

