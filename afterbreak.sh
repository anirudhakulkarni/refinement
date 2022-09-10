export CUDA_VISIBLE_DEVICES='0'
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss LogitNorm
