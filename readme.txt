rsync -avh --update ./ cs5190421@hpc:~/scratch/COD/MDCA-Calibration/ --exclude data --exclude checkpoint --exclude checkpoints 

python3 test.py --model resnet56 --dataset cifar10 --checkpoint checkpoints/cifar10/resnet56_brier_score.pth 


python train.py --dataset mendley --model resnet56 --schedule-steps 80 120 --epochs 160 --loss NLL+MDCA --beta 1.0
python train.py --dataset mnist --model resnet56 --schedule-steps 80 120 --epochs 160 --loss NLL+MDCA --beta 1.0

python3 test.py --model resnet56 --dataset svhn --checkpoint checkpoints/svhn/resnet56_cross_entropy.pth 


installation
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111




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