import os
import argparse
import math

def get_model_name(opt):
    model_name = '{}_{}_{}_{}_im_{}_lr_{}_bsz_{}_g_{}_m_{}_stages_{}'.\
    format(opt.cls_type, opt.loss, opt.dataset, opt.model, opt.imratio, opt.learning_rate,
            opt.batch_size, opt.gamma, opt.margin, opt.stages)
     
    return model_name

def get_num_classes(opt):
    if opt.cls_type == 'binary' and 'auc' in opt.loss :
        return 1
    if opt.cls_type == 'binary':
        return 2
    elif 'cifar100' in opt.dataset :
        return 100
    if 'cifar10' in opt.dataset :
        return 10
    elif 'imagenet' in opt.dataset:
        return 1000
    else:
        raise ValueError('Unknown dataset: {}'.format(opt.dataset))
    
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--learning_rate2', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,75',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--loss', type=str, default='aucm', choices=['ce', 'sls','focal','aucm','aucs'])
    # TODO: List supported models here
    parser.add_argument('--model', type=str, default='resnet50')
    # TODO: List supported datasets here
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--imratio', type=float, default=0.1,
                        help='imbalance ratio for binary classification, Imbalance factor for Long tail dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--margin', type=float, default=1.0, help='margin for AUCM loss')
    parser.add_argument('--gamma', type=float, default=1000, help='gamma for focal loss and AUCM loss')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=123, help='seed for random number')
    # classification type binary or multi-class
    parser.add_argument('--cls_type', type=str, default='binary', choices=['binary', 'multi'],
                        help='classification type: binary or multi-class')
    # 2 stage training. parse initial large epochs and then frequency of small epochs
    parser.add_argument('--stages', type=str, default='1000, 20',
                        help='2 stage training. parse initial large epochs and then frequency of small epochs')
    opt = parser.parse_args()


    opt = update_option(opt)
    return opt


def update_option(opt):
    # set the path according to the environment
    opt.data_folder = '../../data/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = get_model_name(opt)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_cls = get_num_classes(opt)
    
    return opt


