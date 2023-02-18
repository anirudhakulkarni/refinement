from networks.main import SupCEResNet, SupAUCMResNet, SupConResNet
import torch
from .losses import SupConLoss, FocalLoss, AUCMLoss
model_dict = {
    # TODO: Fill these networks
    'ce' : SupCEResNet,
    'focal' : SupCEResNet,
    'sls' : SupConResNet,
    'aucm' : SupAUCMResNet,
    'aucs' : SupAUCMResNet,
}

loss_dict = {
    # TODO: Fill these Lossses
    'ce' : torch.nn.CrossEntropyLoss,
    'sls' : SupConLoss,
    'focal' : FocalLoss,
    'aucm' : AUCMLoss,
    'aucs' : AUCMLoss,
}
def set_model(opt):
    model = model_dict[opt.loss](name=opt.model, num_classes=opt.n_cls)
    if opt.loss == 'ce':
        criterion = loss_dict[opt.loss]()
    elif opt.loss == 'sls':
        criterion = torch.nn.CrossEntropyLoss()
        criterion2 = loss_dict[opt.loss]()
    elif opt.loss == 'focal':
        criterion = loss_dict[opt.loss](gamma=opt.gamma,alpha=opt.alpha)
    elif opt.loss == 'aucm':
        criterion = loss_dict[opt.loss](margin=opt.margin)
    elif opt.loss == 'aucs':
        criterion = loss_dict[opt.loss](margin=1.0)
    else:
        raise ValueError('Loss not supported: {}'.format(opt.loss))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.loss == 'sls':
            criterion2 = criterion2.cuda()
    
    if opt.loss == 'sls':
        return model, criterion, criterion2    
    return model, criterion, None
    