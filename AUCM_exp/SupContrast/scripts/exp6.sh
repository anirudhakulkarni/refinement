# This is for hyperparameter tuning.
# Tune hyperparameters on 5 epochs and choose best to train for 50 epochs
export CUDA_VISIBLE_DEVICES='4'
screen -dm bash -c  \
"python train.py --cls_type multi \
  --loss aucm --dataset imagenet_lt \
  --model resnet18 \
  --batch_size 64 \
  --learning_rate 0.8 \
  --cosine \
  --epochs 5"

export CUDA_VISIBLE_DEVICES='5'
screen -dm bash -c  \
"python train.py --cls_type multi \
  --loss focal --dataset imagenet_lt \
  --model resnet18 \
  --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --epochs 5"

# actual experiment for 30 epochs

export CUDA_VISIBLE_DEVICES='6,7'
screen -dm bash -c  \
"python train.py --cls_type multi \
  --loss ce --dataset imagenet_lt \
  --model resnet18 \
  --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --epochs 30"

# python main_SLS.py --cls_type multi  --loss ce --dataset imagenet_lt --model resnet18 --batch_size 64 --learning_rate 0.8 --cosine --ckpt save/SupCon/imagenet_lt_models/SupCon_imagenet_lt_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_64_temp_0.1_trial_0_cosine/ckpt_epoch_6.pth --epochs 10

screen -dm bash -c  \
"python main_supcon.py --cls_type multi \
  --dataset imagenet_lt \
  --model resnet18 \
  --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --epochs 30"


export CUDA_VISIBLE_DEVICES='4,5'
screen -dm bash -c  \
"python train.py --cls_type multi \
  --loss focal --dataset imagenet_lt \
  --model resnet18 \
  --batch_size 128 \
  --gamma 5 \
  --alpha 0.75 \
  --margin 1.0 \
  --learning_rate 0.8 \
  --cosine \
  --epochs 30 \
  --no_grid"

# python main_supcon.py --cls_type multi   --dataset imagenet_lt   --model resnet18   --batch_size 128   --learning_rate 0.8   --cosine   --epochs 30
# python main_SLS.py --cls_type multi   --dataset imagenet_lt   --model resnet18   --batch_size 128   --learning_rate 0.8   --cosine   --epochs 30
