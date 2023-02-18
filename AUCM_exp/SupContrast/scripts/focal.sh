# Focal
export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset cifar10 --model densenet121 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset cifar10 --model resnet20 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset cifar100 --model densenet121 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset cifar100 --model resnet20 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='1'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset stl10 --model densenet121 --imratio 0.01 --batch_size 32 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='3'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset stl10 --model resnet20 --imratio 0.01 --batch_size 32 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset c2 --model densenet121 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss focal --dataset c2 --model resnet20 --imratio 0.01 --batch_size 128 --learning_rate 0.0003"

