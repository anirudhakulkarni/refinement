#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar10 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=1
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar10 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=2
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar100 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=3
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar100 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=4
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset c2 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=5
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset c2 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=6
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset stl10 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=7
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset stl10 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=0
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar10 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=1
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar10 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=2
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar100 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=3
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar100 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=4
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset c2 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=5
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset c2 \
  --model densenet121 &

export CUDA_VISIBLE_DEVICES=6
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset stl10 \
  --model resnet20 &

export CUDA_VISIBLE_DEVICES=7
python3 main_nll.py --batch_size 512 \
  --loss focal \
  --gamma 1 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset stl10 \
  --model densenet121 &

