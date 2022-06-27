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