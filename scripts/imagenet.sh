# train tiny-imagenet on cross-entropy

python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss cross_entropy

# to train tiny-imagenet on other methods, refer to scripts/cifar10.sh

# focal loss
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss focal_loss --gamma 3.0

# label smoothing
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss LS --alpha 0.1

# MMCE
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss MMCE --beta 4.0

# DCA
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss NLL+DCA --beta 1.0

# FLSD (gamma=3)
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss FLSD --gamma 3.0

# brier score
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss brier_loss

# NLL+MDCA
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss NLL+MDCA --beta 1.0

# LS+MDCA
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss LS+MDCA --alpha 0.1 --beta 1.0

# FL+MDCA
python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss FL+MDCA --gamma 3.0

