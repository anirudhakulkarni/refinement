# python main_ce.py --batch_size 1024 \
#   --learning_rate 0.8 \
#   --cosine --syncBN


export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar10 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset cifar100 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset stl10 --model resnet18 --imratio 0.01 --batch_size 32 --learning_rate 0.8 --cosine"

export CUDA_VISIBLE_DEVICES='2'
screen -dm bash -c  \
"python train.py --cls_type binary --loss ce --dataset c2 --model resnet18 --imratio 0.01 --batch_size 128 --learning_rate 0.8 --cosine"
