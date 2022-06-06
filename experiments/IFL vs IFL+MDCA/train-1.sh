export CUDA_VISIBLE_DEVICES="4"
python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 0.1

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 1.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 2.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 3.0


