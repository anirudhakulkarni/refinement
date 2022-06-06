# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss cross_entropy

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss NLL+MDCA --beta 1.0
export CUDA_VISIBLE_DEVICES="2,3"
python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 5.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 15.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss NLL+MDCA --beta 20.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss NLL+MDCA --beta 25.0

