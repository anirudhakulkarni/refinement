
# given a training log file show the training curve, accuracy curve, loss curve, ece loss

'''
sample log file:
INFO:  Setting up logging folder : checkpoint/imagenet/26-May_resnet50_imagenet_cross_entropy
INFO:  Using model : resnet50_imagenet
INFO:  Using dataset : imagenet
INFO:  Setting up optimizer : sgd
INFO:  Step sizes : [40, 60] | lr-decay-factor : 0.1
INFO:  Epoch: [1 | 100] LR: 0.100000
INFO:  End of epoch 1 stats: train_loss : 5.4581 | val_loss : 5.1097 | top1_train : 0.9867 | top1 : 2.0100 | SCE : 0.00115 | ECE : 0.00924 | AUROC : 0.675138
INFO:  Epoch: [2 | 100] LR: 0.100000
INFO:  End of epoch 2 stats: train_loss : 4.9602 | val_loss : 4.7977 | top1_train : 2.8422 | top1 : 4.1900 | SCE : 0.00112 | ECE : 0.00989 | AUROC : 0.653346
INFO:  Epoch: [3 | 100] LR: 0.100000
INFO:  End of epoch 3 stats: train_loss : 4.6809 | val_loss : 4.5673 | top1_train : 5.6489 | top1 : 6.8500 | SCE : 0.00192 | ECE : 0.01273 | AUROC : 0.657884
INFO:  Epoch: [4 | 100] LR: 0.100000
'''

def get_basic_curves(logfilename):
    log=open(logfilename,'r')
    lines=log.readlines()
    log.close()
    train_loss=[]
    val_loss=[]
    top1_train=[]
    top1=[]
    sce = []
    ece = []
    auroc = []
    for line in lines:
        if 'INFO:  End of epoch' in line:
            # print(line)
            # print(line.split('train_loss : '))
            train_loss.append(float(line.split('train_loss: ')[1].split(" ")[0]))
            val_loss.append(float(line.split('val_loss: ')[1].split(" ")[0]))
            top1_train.append(float(line.split('top1_train: ')[1].split(" ")[0]))
            top1.append(float(line.split('top1: ')[1].split(" ")[0]))
            sce.append(float(line.split('SCE: ')[1].split(" ")[0]))
            ece.append(float(line.split('ECE: ')[1].split(" ")[0]))
            try:
                auroc.append(float(line.split('AUROC: ')[1].split(" ")[0]))
            except:
                pass
    return train_loss, val_loss, top1_train, top1, sce, ece, auroc
import argparse

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='show training curve')
    parser.add_argument('--logfile', type=str, default='checkpoint/imagenet/26-May_resnet50_imagenet_cross_entropy/train.log', help='log file path')
    args = parser.parse_args()
    train_loss, val_loss, top1_train, top1, sce, ece, auroc = get_basic_curves(args.logfile)
    import matplotlib.pyplot as plt
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(args.logfile+'_loss.png')
    plt.clf()
    plt.plot(top1_train, label='top1_train')
    plt.plot(top1, label='top1_test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(args.logfile+'_top1_traintest.png')
    plt.clf()
    sce=[x*1000 for x in sce]
    ece = [x*100 for x in ece]
    plt.plot(sce, label='sce (10^-3)')
    plt.plot(ece, label='ece (%)')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.savefig(args.logfile+'_metrics.png')
    plt.clf()
    plt.plot(auroc, label='auroc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('auroc')
    plt.savefig(args.logfile+'_auroc.png')