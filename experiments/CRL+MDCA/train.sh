loss	dataset	Beta	gamma	theta		
NLL+CRL	TinyImg	-	-	0.1		
	TinyImg	-	-	0.5		
	TinyImg	-	-	1		
	TinyImg	-	-	3		
NLL+CRL+MDCA	TinyImg	1	-	0.5		
	TinyImg	5	-	0.5		
	TinyImg	10	-	0.5		
	TinyImg	1	- 	1		
	TinyImg	5	- 	1		
	TinyImg	10	- 	1		
	TinyImg	1	-	3		
	TinyImg	5	- 	3		
	TinyImg	10	- 	3		
FL+CRL+MDCA	TinyImg	5	1	3		
	TinyImg	5	3	3		

python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 0.1
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 0.5
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 1
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL --theta 3

python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 0.5 --beta 1.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 0.5 --beta 5.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 0.5 --beta 10.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 1.0 --beta 1.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 1.0 --beta 5.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 1.0 --beta 10.0

python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 1.0 --beta 1.0 --gamma 1.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 1.0 --beta 1.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 3.0 --beta 1.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 3.0 --beta 5.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss NLL+CRL+MDCA --theta 3.0 --beta 10.0

# gpu 0,1
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 1.0 --beta 5.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 0.5 --beta 5.0 --gamma 3.0


python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 2.0 --beta 5.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCA --theta 3.0 --beta 5.0 --gamma 3.0


# before 9th there was a bug in code whihc resulted theta value being fixed to 0.25. Hence all models before taht corresponds to theta=0.25.
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCAexp --theta 1.0 --beta 10.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCAcubic --theta 1.0 --beta 10.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCAsquare --theta 1.0 --beta 10.0 --gamma 3.0
python train.py --dataset imagenet_CRL --model resnet50_imagenet --epochs 160 --schedule-steps 80 120 --loss FL+CRL+MDCAsmooth --theta 1.0 --beta 10.0 --gamma 3.0
