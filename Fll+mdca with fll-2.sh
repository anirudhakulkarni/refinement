
# svhn
export CUDA_VISIBLE_DEVICES="6,7"

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 1.0


# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 2.0

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 3.0

# # gamma=3.0

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 1.0

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 5.0

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 10.0

# python train.py \
# --dataset svhn \
# --model resnet56 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 20.0

# gamma=2.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 1.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 5.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 10.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 20.0

# gamma=1.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 1.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 5.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 10.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 20.0

