# gama in range 1,2,3
# alpha in range 1, 5, 10, 15, 20, 25


# first for focal loss in range 1,2,3
export CUDA_VISIBLE_DEVICES="4,5"
# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 1.0 

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 2.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss focal_loss --gamma 3.0

# # focal loss with mdca

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 1.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 5.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 10.0

# python train.py \
# --dataset cifar10 \
# --model resnet32 \
# --schedule-steps 80 120 \
# --epochs 160 \
# --loss FL+MDCA --gamma 3.0 --beta 20.0

# gamma = 1.0


python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 1.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 5.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 20.0

# gamma = 2.0


python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 1.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 5.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 10.0

python train.py \
--dataset cifar10 \
--model resnet32 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 2.0 --beta 20.0

### corresponding testing commands
# first for focal loss in range 1,2,3
# checkpoint/cifar10/29-May_resnet32_focal_loss_gamma=1.0_anirudha
'''
'29-May_resnet32_FL+MDCA_gamma=1.0_beta=10.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=1.0_beta=1.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=1.0_beta=5.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=3.0_beta=10.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=3.0_beta=1.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=3.0_beta=20.0_anirudha'
'29-May_resnet32_FL+MDCA_gamma=3.0_beta=5.0_anirudha'
'29-May_resnet32_focal_loss_gamma=1.0_anirudha'
'29-May_resnet32_focal_loss_gamma=2.0_anirudha'
'29-May_resnet32_focal_loss_gamma=3.0_anirudha'
'30-May_resnet32_FL+MDCA_gamma=1.0_beta=20.0_anirudha'
'30-May_resnet32_FL+MDCA_gamma=2.0_beta=10.0_anirudha'
'30-May_resnet32_FL+MDCA_gamma=2.0_beta=1.0_anirudha'
'30-May_resnet32_FL+MDCA_gamma=2.0_beta=20.0_anirudha'
'30-May_resnet32_FL+MDCA_gamma=2.0_beta=5.0_anirudha'
'''

# focal loss with mdca
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=1.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=1.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=1.0_beta=5.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=3.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=3.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=3.0_beta=20.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_FL+MDCA_gamma=3.0_beta=5.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_focal_loss_gamma=1.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_focal_loss_gamma=2.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/29-May_resnet32_focal_loss_gamma=3.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/30-May_resnet32_FL+MDCA_gamma=1.0_beta=20.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/30-May_resnet32_FL+MDCA_gamma=2.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/30-May_resnet32_FL+MDCA_gamma=2.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/30-May_resnet32_FL+MDCA_gamma=2.0_beta=20.0_anirudha/model_best.pth
python3 test.py --model resnet32 --dataset cifar10 --checkpoint checkpoint/cifar10/30-May_resnet32_FL+MDCA_gamma=2.0_beta=5.0_anirudha/model_best.pth



'29-May_resnet56_FL+MDCA_gamma=2.0_beta=1.0_anirudha'
'29-May_resnet56_FL+MDCA_gamma=2.0_beta=5.0_anirudha'
'29-May_resnet56_FL+MDCA_gamma=3.0_beta=10.0_anirudha'
'29-May_resnet56_FL+MDCA_gamma=3.0_beta=1.0_anirudha'
'29-May_resnet56_FL+MDCA_gamma=3.0_beta=20.0_anirudha'
'29-May_resnet56_FL+MDCA_gamma=3.0_beta=5.0_anirudha'
'29-May_resnet56_focal_loss_gamma=1.0_anirudha'
'29-May_resnet56_focal_loss_gamma=2.0_anirudha'
'29-May_resnet56_focal_loss_gamma=3.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=1.0_beta=10.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=1.0_beta=1.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=1.0_beta=20.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=1.0_beta=5.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=2.0_beta=10.0_anirudha'
'30-May_resnet56_FL+MDCA_gamma=2.0_beta=20.0_anirudha'

python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=2.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=2.0_beta=5.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=3.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=3.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=3.0_beta=20.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_FL+MDCA_gamma=3.0_beta=5.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_focal_loss_gamma=1.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_focal_loss_gamma=2.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/29-May_resnet56_focal_loss_gamma=3.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=1.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=1.0_beta=1.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=1.0_beta=20.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=1.0_beta=5.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=2.0_beta=10.0_anirudha/model_best.pth
python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoint/svhn/30-May_resnet56_FL+MDCA_gamma=2.0_beta=20.0_anirudha/model_best.pth