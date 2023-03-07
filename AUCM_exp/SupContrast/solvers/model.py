from networks.main import SupCEResNet, SupAUCMResNet, SupConResNet, LinearClassifier, Cifar100IMBModel, resnet18
import torch
from solvers.losses import DCA, LogitNormLoss

from utils.util import is_CE_like
from .losses import SupConLoss, FocalLoss, AUCMLoss, ClassficationAndMDCA, InverseFocalLoss, LabelSmoothingLoss, BrierScore
import torch.backends.cudnn as cudnn
model_dict = {
    # TODO: Fill these networks
    'ce' : SupCEResNet,
    'focal' : SupCEResNet,
    'ifl' : SupCEResNet,
    'supcon' : SupConResNet,
    'aucm' : SupAUCMResNet,
    'aucs' : SupAUCMResNet,
    'ce_linear': LinearClassifier,
    'mdca_linear': LinearClassifier,
}
multi_model_dict = {
    'ce' : Cifar100IMBModel,
    'aucm' : Cifar100IMBModel,
}

loss_dict = {
    # TODO: Fill these Lossses
    'ce' : torch.nn.CrossEntropyLoss,
    'supcon' : SupConLoss,
    'focal' : FocalLoss,
    'aucm' : AUCMLoss,
    'aucs' : AUCMLoss,
    'ce_linear': torch.nn.CrossEntropyLoss,
    'mdca_linear': ClassficationAndMDCA,
    'ifl': InverseFocalLoss,
    'ls': LabelSmoothingLoss,
    'logitnorm': LogitNormLoss,
    'brier': BrierScore,
    'dca': DCA
}
def set_model(opt):
    if opt.cls_type == 'multi':
        if opt.dataset == 'cifar100_imb':
            model = multi_model_dict[opt.loss](name=opt.model, num_classes=opt.n_cls)    
        else:
            # imagenet_lt
            model = torch.nn.Sequential(resnet18(), torch.nn.Linear(512, opt.n_cls))
        if opt.loss == 'ce':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'supcon':
            criterion = loss_dict[opt.loss](temperature=opt.temp)
        elif opt.loss == 'focal':
            criterion = loss_dict[opt.loss](gamma=opt.gamma,alpha=opt.alpha)
        elif opt.loss == 'aucm':
            criterion = loss_dict[opt.loss](margin=opt.margin)
        elif opt.loss == 'aucs':
            criterion = loss_dict[opt.loss](margin=1.0)
        elif opt.loss == 'ce_linear':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'mdca_linear':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'ifl':
            criterion = loss_dict[opt.loss](gamma=opt.gamma)
        else:
            raise ValueError('Loss not supported: {}'.format(opt.loss))
    else:
        if opt.loss == 'supcon':
            model = model_dict[opt.loss](name=opt.model)
        elif is_CE_like(opt.loss):
            model = model_dict['ce'](name=opt.model, num_classes=opt.n_cls)
        else:
            model = model_dict[opt.loss](name=opt.model, num_classes=opt.n_cls)
        
        if opt.loss == 'ce':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'supcon':
            criterion = loss_dict[opt.loss](temperature=opt.temp)
        elif opt.loss == 'focal':
            criterion = loss_dict[opt.loss](gamma=opt.gamma,alpha=opt.alpha)
        elif opt.loss == 'aucm':
            criterion = loss_dict[opt.loss](margin=opt.margin)
        elif opt.loss == 'aucs':
            criterion = loss_dict[opt.loss](margin=1.0)
        elif opt.loss == 'ce_linear':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'mdca_linear':
            criterion = loss_dict[opt.loss]()
        elif opt.loss == 'ifl':
            criterion = loss_dict[opt.loss](gamma=opt.gamma)
        elif opt.loss == 'ls':
            criterion = loss_dict[opt.loss](alpha=0.1)
        elif opt.loss == 'dca' or opt.loss == 'brier' or opt.loss == 'logitnorm':
            criterion = loss_dict[opt.loss]()
        else:
            raise ValueError('Loss not supported: {}'.format(opt.loss))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion
    
def set_model_linear(opt):
    model = SupConResNet(name=opt.model)
    
    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion1, criterion2

