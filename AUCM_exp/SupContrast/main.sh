

python main_linear.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_350.pth 

python main_ce.py --batch_size 512 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1

python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 

python main_supcon.py --batch_size 256 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --imratio 0.1



# 3, 2658302.pts-45.user: supcon
# !5, 2667884.pts-45.user: aucm
# !6, 2718616.pts-45.user: CE
# 7, : linear

# change dataset to cifar100 and C2
# vacant gpus: 1, 2, 4, 6, 7

# !7: ce, c2         2852982.pts-45.user
# !6: aucm, c2       2889050.pts-45.user
# 4: supcon, c2   
# !2: ce, cifar100   2892413.pts-45.user
# !7: aucm, cifar100 2909642.pts-45.user

python3 main_ce.py --batch_size 512 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset c2


python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset c2

python main_supcon.py --batch_size 128 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --imratio 0.1 \
  --dataset c2



python3 main_ce.py --batch_size 512 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar100


python main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset cifar100


# ! 3172778.pts-45.user
python3 main_ce.py --batch_size 512 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset stl10

# ! 3195219.pts-45.user
python3 main_aucm.py --batch_size 128 \
  --learning_rate 0.8 \
  --cosine \
  --imratio 0.1 \
  --dataset stl10

python main_supcon.py --batch_size 128   --learning_rate 0.5   --temp 0.1   --cosine   --imratio 0.1   --dataset stl10

python main_linear.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset stl10 \
  --ckpt ./save/SupCon/stl10_models/SupCon_stl10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth 

python main_linear.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset c2 \
  --ckpt ./save/SupCon/c2_models/SupCon_c2_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth

python main_supcon.py --batch_size 256   --learning_rate 1   --temp 0.1   --cosine   --imratio 0.1   --dataset cifar100


# layers not frozen [detached from 1735059.pts-32.user]
python main_linear_unfrz.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset stl10 \
  --ckpt ./save/SupCon/stl10_models/SupCon_stl10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth 

python main_linear_unfrz.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset cifar10 \
  --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_350.pth 

python main_linear_unfrz.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset c2 \
  --ckpt ./save/SupCon/c2_models/SupCon_c2_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth

python main_linear_unfrz.py --batch_size 256 \
  --learning_rate 5 \
  --imratio 0.1 \
  --dataset cifar100 \
  --ckpt ./save/SupCon/cifar100_models/SupCon_cifar100_resnet50_im_0.1_lr_1.0_decay_0.0001_bsz_256_temp_0.07_trial_0/ckpt_epoch_1000.pth


# export CUDA_VISIBLE_DEVICES=4
#  1797262.pts-32.user
python main_SLS.py --batch_size 128 \
  --learning_rate 0.8  \
  --imratio 0.1 \
  --dataset stl10 \
  --ckpt ./save/SupCon/stl10_models/SupCon_stl10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth


# export CUDA_VISIBLE_DEVICES=5
# 1915304.pts-28.user
python main_SLS.py --batch_size 128 \
  --learning_rate 0.8  \
  --imratio 0.1 \
  --dataset cifar10 \
  --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth 

# export CUDA_VISIBLE_DEVICES=6
# 1917305.pts-28.user
python main_SLS.py --batch_size 128 \
  --learning_rate 0.8  \
  --imratio 0.1 \
  --dataset c2 \
  --ckpt ./save/SupCon/c2_models/SupCon_c2_resnet50_im_0.1_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth
