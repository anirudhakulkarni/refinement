export CUDA_VISIBLE_DEVICES="4"
python train.py \
--dataset cifar100 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 0.1

python train.py \
--dataset cifar100 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 1.0

python train.py \
--dataset cifar100 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 2.0

python train.py \
--dataset cifar100 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 3.0




export CUDA_VISIBLE_DEVICES='0'
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss cross_entropy 
export CUDA_VISIBLE_DEVICES='1' 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 10.0 
export CUDA_VISIBLE_DEVICES='2' 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 
python train.py --dataset cifar100 --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA --gamma 3.0 --beta 5.0 
export CUDA_VISIBLE_DEVICES='3' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 
export CUDA_VISIBLE_DEVICES='4' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRL --theta 1.0 --beta 5.0 --gamma 3.0 --pairing complete 
export CUDA_VISIBLE_DEVICES='5' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLsmooth --theta 1.0 --beta 5.0 --gamma 3.0  
export CUDA_VISIBLE_DEVICES='6' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.5 
export CUDA_VISIBLE_DEVICES='0' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 2.0 

export CUDA_VISIBLE_DEVICES='1' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 5.0 --gamma 3.0 --scalefactor 1.25 

export CUDA_VISIBLE_DEVICES='6' 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 
python train.py --dataset cifar100_CRL --model resnet56 --epochs 160 --schedule-steps 80 120 --loss FL+MDCA+CRLscale --theta 1.0 --beta 0 --gamma 3.0 --scalefactor 1.25 


export CUDA_VISIBLE_DEVICES='7' 
python train.py --dataset cifar10_CRL --model resnet32 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0
python train.py --dataset cifar10_CRL --model resnet32 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0
python train.py --dataset cifar10_CRL --model resnet32 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0
python train.py --dataset cifar10_CRL --model resnet32 --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1.0
