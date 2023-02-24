# export CUDA_VISIBLE_DEVICES='4' 
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar10 --imratio 0.01 --model resnet18 --cosine"

# export CUDA_VISIBLE_DEVICES='5'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --imratio 0.01 --model resnet18 --cosine"

# export CUDA_VISIBLE_DEVICES='6'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset stl10 --imratio 0.01 --model resnet18 --cosine"

# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.01 --model resnet18 --cosine"



export CUDA_VISIBLE_DEVICES='0' 
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar10 --imratio 0.1 --model resnet18 --cosine"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --imratio 0.1 --model resnet18 --cosine"

export CUDA_VISIBLE_DEVICES='6'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset stl10 --imratio 0.1 --model resnet18 --cosine"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python main_supcon.py --batch_size 1024   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.1 --model resnet18 --cosine"


# resnet50
# export CUDA_VISIBLE_DEVICES='6'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset cifar10 --imratio 0.01 --model resnet50 --cosine"

# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --imratio 0.01 --model resnet50 --cosine"

# export CUDA_VISIBLE_DEVICES='6'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset stl10 --imratio 0.01 --model resnet50 --cosine"

# export CUDA_VISIBLE_DEVICES='7'
# screen -dm bash -c  \
# "python main_supcon.py --batch_size 512   --learning_rate 0.5   --temp 0.1 --dataset c2 --imratio 0.01 --model resnet50 --cosine"