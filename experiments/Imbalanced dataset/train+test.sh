

python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss cross_entropy

python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss NLL+MDCA --beta 1
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss NLL+MDCA --beta 5
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss NLL+MDCA --beta 10
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss NLL+MDCA --beta 15
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss NLL+MDCA --beta 20

python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss focal_loss --gamma 1.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 1 --gamma 1.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 5 --gamma 1.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 10 --gamma 1.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 15 --gamma 1.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 20 --gamma 1.0

python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss focal_loss --gamma 2.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 1 --gamma 2.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 5 --gamma 2.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 10 --gamma 2.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 15 --gamma 2.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 20 --gamma 2.0

python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss focal_loss --gamma 3.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 1 --gamma 3.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 5 --gamma 3.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 10 --gamma 3.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 15 --gamma 3.0
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss FL+MDCA --beta 20 --gamma 3.0


python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS  --alpha 0.01
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS+MDCA --beta 1 --alpha 0.01
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS+MDCA --beta 5 --alpha 0.01
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS+MDCA --beta 10 --alpha 0.01
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS+MDCA --beta 15 --alpha 0.01
python train_imbalanced.py --dataset im_cifar10 --model resnet32 --schedule-steps 80 120 --epochs 160 --imbalance 0.01 --loss LS+MDCA --beta 20 --alpha 0.01


