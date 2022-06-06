export CUDA_VISIBLE_DEVICES="4"
python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS --alpha 0.1

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 1.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 5.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 15.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 20.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 25.0
