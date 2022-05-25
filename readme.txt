python3 test.py --model resnet56 --dataset cifar10 --checkpoint MDCA-checkpoints/cifar10/resnet56_brier_score.pth 


python train.py --dataset mendley --model resnet56 --schedule-steps 80 120 --epochs 160 --loss NLL+MDCA --beta 1.0
python train.py --dataset mnist --model resnet56 --schedule-steps 80 120 --epochs 160 --loss NLL+MDCA --beta 1.0

python3 test.py --model resnet56 --dataset svhn --checkpoint MDCA-checkpoints/svhn/resnet56_cross_entropy.pth 


installation
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111