# AUCM
export CUDA_VISIBLE_DEVICES='4' 
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss aucm --dataset cifar10 --model resnet50 --imratio 0.1 --batch_size 127 --learning_rate 0.1 --lr_decay_epochs 300,600,800,900 &&  \
python3 train.py --cls_type binary --loss aucm --dataset cifar100 --model resnet50 --imratio 0.1 --batch_size 127 --learning_rate 0.1 --lr_decay_epochs 300,600,800,900 && \
python3 train.py --cls_type binary --loss aucm --dataset stl10 --model resnet50 --imratio 0.1 --batch_size 32 --learning_rate 0.1 --lr_decay_epochs 300,600,800,900 && \
python3 train.py --cls_type binary --loss aucm --dataset c2 --model resnet50 --imratio 0.1 --batch_size 128 --learning_rate 0.1 --lr_decay_epochs 300,600,800,900"

