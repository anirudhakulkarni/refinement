export CUDA_VISIBLE_DEVICES="5"
python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss cross_entropy

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 1.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 5.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 10.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 15.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 20.0

python train.py \
--dataset svhn \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 25.0


python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 25.0

python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss NLL+MDCA --beta 15.0


###################################


export CUDA_VISIBLE_DEVICES="6"
python train.py \
--dataset svhn \
--model resnet20 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 5.0

python train.py \
--dataset svhn \
--model resnet20 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 5.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 40 60 \
--epochs 100 \
--loss NLL+MDCA --beta 15.0


python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss NLL+MDCA --beta 20.0

