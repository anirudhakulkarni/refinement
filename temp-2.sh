export CUDA_VISIBLE_DEVICES='0'

screen bash -c \
"python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
"
export CUDA_VISIBLE_DEVICES='1' 
screen bash -c \
"python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 
"
export CUDA_VISIBLE_DEVICES='2' 
screen bash -c \
"python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 && \
python train.py --dataset svhn --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 
"
export CUDA_VISIBLE_DEVICES='3' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 
"
export CUDA_VISIBLE_DEVICES='4' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete 
"
export CUDA_VISIBLE_DEVICES='5' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  
"
export CUDA_VISIBLE_DEVICES='6' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 && \ 
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 
"
export CUDA_VISIBLE_DEVICES='2' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 
"
export CUDA_VISIBLE_DEVICES='4' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 
"
export CUDA_VISIBLE_DEVICES='7' 
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 
"
export CUDA_VISIBLE_DEVICES='0'
screen bash -c \
"python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset svhn_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0
"

export CUDA_VISIBLE_DEVICES='4,5,6'
screen bash -c \
"python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm 
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm 
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm 
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm 
"

export CUDA_VISIBLE_DEVICES='3'
screen bash -c \
"python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
"

export CUDA_VISIBLE_DEVICES='2'
screen bash -c \
"python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
"


export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
"



export CUDA_VISIBLE_DEVICES='2,3,7'
screen bash -c \
"python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm+MDCA --beta 20.0
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm+MDCA --beta 20.0
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm+MDCA --beta 20.0
python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss LogitNorm+MDCA --beta 20.0
"


export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --dataset cifar100 --model resnet32 --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model resnet32 --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model resnet32 --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset cifar100 --model resnet32 --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
"


export CUDA_VISIBLE_DEVICES='0'
screen bash -c \
"python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm+MDCA --beta 5.0
"
export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
"
export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset svhn --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
"

python posthoc_calibrate.py --checkpoint checkpoint/imagenet/04-Jul_resnet50_imagenet_LogitNorm_adjacent_anirudha/model_best.pth --model resnet50_imagenet --dataset imagenet
python posthoc_calibrate.py --checkpoint checkpoint/imagenet/05-Jul_resnet50_imagenet_LogitNorm+MDCA_adjacent_beta=5.0_anirudha/model_best.pth --model resnet50_imagenet --dataset imagenet
python posthoc_calibrate.py --checkpoint checkpoint/cifar10/04-Jul_resnet50_imagenet_LogitNorm_adjacent_anirudha/model_best.pth --model resnet32 --dataset cifar10
python posthoc_calibrate.py --checkpoint checkpoint/svhn/05-Jul_resnet50_imagenet_LogitNorm+MDCA_adjacent_anirudha/model_best.pth --model wrn --dataset svhn

export CUDA_VISIBLE_DEVICES='4,5,6'
screen bash -c \
"python3 train.py --dataset imagenet --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss cross_entropy
"

export CUDA_VISIBLE_DEVICES='1'
screen bash -c \
"python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss LogitNorm
"

export CUDA_VISIBLE_DEVICES='2'
screen bash -c \
"python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA --beta 10.0 --gamma 3.0
"

export CUDA_VISIBLE_DEVICES='3'
screen bash -c \
"python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+MDCA --beta 10.0
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+MDCA --beta 10.0 
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+MDCA --beta 10.0 
python3 train.py --dataset cifar10 --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+MDCA --beta 10.0 
"
export CUDA_VISIBLE_DEVICES='4'
screen bash -c \
"python3 train.py --dataset cifar10_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset cifar10_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
"

export CUDA_VISIBLE_DEVICES='6'
screen bash -c \
"python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
python3 train.py --dataset cifar100 --model wrn --epochs 200 --schedule-steps 80 140 --loss cross_entropy
"


export CUDA_VISIBLE_DEVICES='4'
screen bash -c \
"python3 train.py --dataset svhn_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset svhn_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset svhn_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
python3 train.py --dataset svhn_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss FL+MDCA+CRL --beta 10.0 --gamma 3.0
"

export CUDA_VISIBLE_DEVICES='3'
screen bash -c \
"python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0
"
export CUDA_VISIBLE_DEVICES='3'
screen bash -c \
"python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0 && \
python train.py --dataset cifar100_CRL --model wrn --epochs 200 --schedule-steps 80 140 --loss NLL+CRL --theta 1.0
"
