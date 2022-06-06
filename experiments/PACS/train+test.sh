python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss cross_entropy

python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss NLL+MDCA --beta 1
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss NLL+MDCA --beta 5
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss NLL+MDCA --beta 10
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss NLL+MDCA --beta 15
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss NLL+MDCA --beta 20

python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss focal_loss --gamma 1.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 1 --gamma 1.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 5 --gamma 1.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 10 --gamma 1.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 15 --gamma 1.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 20 --gamma 1.0

python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss focal_loss --gamma 2.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 1 --gamma 2.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 5 --gamma 2.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 10 --gamma 2.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 15 --gamma 2.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 20 --gamma 2.0

python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss focal_loss --gamma 3.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 1 --gamma 3.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 5 --gamma 3.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 10 --gamma 3.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 15 --gamma 3.0
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss FL+MDCA --beta 20 --gamma 3.0


python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS  --alpha 0.01
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS+MDCA --beta 1 --alpha 0.01
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS+MDCA --beta 5 --alpha 0.01
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS+MDCA --beta 10 --alpha 0.01
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS+MDCA --beta 15 --alpha 0.01
python train.py --dataset pacs --model resnet_pacs --train-batch-size 256 --test-batch-size 256 --epochs 30 --schedule-steps 20 --lr 0.01 --loss LS+MDCA --beta 20 --alpha 0.01




# test
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_cross_entropy_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_NLL+MDCA_beta\=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_NLL+MDCA_beta\=5.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_NLL+MDCA_beta\=10.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_NLL+MDCA_beta\=15.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_NLL+MDCA_beta\=20.0_anirudha/model_best.pth

python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_focal_loss_gamma\=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=1.0_beta\=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=1.0_beta\=5.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=1.0_beta\=10.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=1.0_beta\=15.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=1.0_beta\=20.0_anirudha/model_best.pth

python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_focal_loss_gamma\=2.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=2.0_beta\=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=2.0_beta\=5.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=2.0_beta\=10.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=2.0_beta\=15.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=2.0_beta\=20.0_anirudha/model_best.pth

python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_focal_loss_gamma\=3.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=3.0_beta\=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=3.0_beta\=5.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=3.0_beta\=10.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=3.0_beta\=15.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_FL+MDCA_gamma\=3.0_beta\=20.0_anirudha/model_best.pth

python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS_alpha=0.01_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS+MDCA_alpha=0.01_beta=1.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS+MDCA_alpha=0.01_beta=5.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS+MDCA_alpha=0.01_beta=10.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS+MDCA_alpha=0.01_beta=15.0_anirudha/model_best.pth
python ood_pacs.py --dataset pacs --model resnet_pacs --checkpoint checkpoint/pacs/03-Jun_resnet_pacs_LS+MDCA_alpha=0.01_beta=20.0_anirudha/model_best.pth