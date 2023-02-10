# table 1
## ResNet20
### AUCM
export CUDA_VISIBLE_DEVICES=4
python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar10 \
  --model resnet20 &&

python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar100 \
  --model resnet20 &&

python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset c2 \
  --model resnet20 &&


python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset stl10 \
  --model resnet20 &

## DenseNet121
### AUCM
export CUDA_VISIBLE_DEVICES=5
python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar10 \
  --model densenet121 &&

python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset cifar100 \
  --model densenet121 &&

python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset c2 \
  --model densenet121 &&

python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.01 \
  --dataset stl10 \
  --model densenet121 &




# Table 8
## ResNet20
### AUCM
export CUDA_VISIBLE_DEVICES=6
python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar10 \
  --model resnet20 &&

python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar100 \
  --model resnet20 &&

python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset c2 \
  --model resnet20 &&

python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset stl10 \
  --model resnet20 &

## DenseNet121
### AUCM
export CUDA_VISIBLE_DEVICES=7
python main_aucm.py --batch_size 128 \
    --learning_rate 0.8 \
    --cosine \
    --imratio 0.1 \
    --dataset cifar10 \
    --model densenet121 &&

python main_aucm.py --batch_size 128 \
    --learning_rate 0.8 \
    --cosine \
    --imratio 0.1 \
    --dataset cifar100 \
    --model densenet121 &&

python3 main_aucm.py --batch_size 128 \
    --learning_rate 0.8 \
    --cosine \
    --imratio 0.1 \
    --dataset c2 \
    --model densenet121 &&

python3 main_aucm.py --batch_size 128 \
    --learning_rate 0.8 \
    --cosine \
    --imratio 0.1 \
    --dataset stl10 \
    --model densenet121





