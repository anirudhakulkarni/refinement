
# SVHN

export CUDA_VISIBLE_DEVICES="5"
python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 0.1

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 1.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 2.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss IFL --gamma 3.0

