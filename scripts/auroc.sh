export CUDA_VISIBLE_DEVICES="2"
# cifar10
python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss cross_entropy

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 1.0

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

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 20.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 25.0


python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_brier_score.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_cross_entropy.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_FL+MDCA.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_FLSD.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_focal_loss.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_LS+MDCA.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_LS.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_MMCE.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_NLL+DCA.pth
python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_NLL+MDCA.pth


python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_brier_score.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_cross_entropy.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_FL+MDCA.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_FLSD.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_focal_loss.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_LS+MDCA.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_LS.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_MMCE.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_NLL+DCA.pth
python3 test.py --model resnet56 --dataset cifar100 --checkpoint checkpoints/cifar100/resnet56_NLL+MDCA.pth


python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_brier_score.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_cross_entropy.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_FL+MDCA.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_FLSD.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_focal_loss.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_LS+MDCA.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_LS.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_MMCE.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_NLL+DCA.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_NLL+MDCA.pth