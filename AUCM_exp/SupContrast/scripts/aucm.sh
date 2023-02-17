# Table 1: Testing AUC on benchmark datasets with imratio=1%.
# Dataset CE Focal AUC-S AUC-M
# C2 (D) 0.718±0.018 0.713±0.009 0.803±0.018 0.809±0.016
# C10 (D) 0.698±0.017 0.700±0.007 0.745±0.010 0.760±0.006
# S10 (D) 0.641±0.032 0.660±0.027 0.669±0.070 0.703±0.030
# C100 (D) 0.588±0.011 0.591±0.017 0.607±0.010 0.614±0.016
# C2 (R) 0.730±0.028 0.724±0.020 0.748±0.007 0.756±0.017
# C10 (R) 0.690±0.011 0.681±0.011 0.702±0.015 0.715±0.008
# S10 (R) 0.641±0.021 0.634±0.024 0.645±0.029 0.659±0.020
# C100 (R) 0.563±0.015 0.565±0.022 0.587±0.017 0.596±0.016

export CUDA_VISIBLE_DEVICES='0' 
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model densenet121 --imratio 0.01 --batch_size 128" 

export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model resnet20 --imratio 0.01 --batch_size 128"

export CUDA_VISIBLE_DEVICES='2'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model densenet121 --imratio 0.01 --batch_size 128"

export CUDA_VISIBLE_DEVICES='3'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model resnet20 --imratio 0.01 --batch_size 128"

export CUDA_VISIBLE_DEVICES='4'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset stl10 --model densenet121 --imratio 0.01 --batch_size 32"

export CUDA_VISIBLE_DEVICES='5'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset stl10 --model resnet20 --imratio 0.01 --batch_size 32"

export CUDA_VISIBLE_DEVICES='6'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset c2 --model densenet121 --imratio 0.01 --batch_size 128"

export CUDA_VISIBLE_DEVICES='7'
screen bash -c \
"python3 train.py --cls_type binary --loss aucm --dataset c2 --model resnet20 --imratio 0.01 --batch_size 128"

