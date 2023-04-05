export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.8"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python train.py --cls_type binary --loss focal --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8"
