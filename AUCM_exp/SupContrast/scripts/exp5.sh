export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss focal --dataset cifar10  --model resnet18  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss focal --dataset c2   --model resnet18 --ckpt save/SupCon/c2_models/SupCon_c2_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss focal --dataset cifar100   --model resnet18 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 32   --learning_rate 5   --imratio 0.1  --loss focal --dataset stl10   --model resnet18 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth
"

export CUDA_VISIBLE_DEVICES='7'
screen -dm bash -c  \
"python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss mdca --dataset cifar10  --model resnet18  --ckpt save/SupCon/cifar10_models/SupCon_cifar10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss mdca --dataset c2   --model resnet18 --ckpt save/SupCon/c2_models/SupCon_c2_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 128   --learning_rate 5   --imratio 0.1  --loss mdca --dataset cifar100   --model resnet18 --ckpt save/SupCon/cifar100_models/SupCon_cifar100_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth && \
python main_SLS.py --batch_size 32   --learning_rate 5   --imratio 0.1  --loss mdca --dataset stl10   --model resnet18 --ckpt save/SupCon/stl10_models/SupCon_stl10_resnet18_im_0.1_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_1000.pth
"