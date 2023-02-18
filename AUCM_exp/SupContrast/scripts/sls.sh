export CUDA_VISIBLE_DEVICES='0'
screen -dm bash -c  \
"python3 train.py --cls_type binary --loss sls --dataset cifar10 --model densenet121 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --learning_rate2 0.0003"

# python3 train.py --cls_type binary --loss sls --dataset cifar10 --model resnet20 --imratio 0.01 --batch_size 128 --learning_rate 0.1 --learning_rate2 0.0003