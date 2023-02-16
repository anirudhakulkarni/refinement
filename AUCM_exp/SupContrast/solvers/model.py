from networks.main import SupCEResNet, SupAUCMResNet
import torch
from losses import SupConLoss, FocalLoss, AUCMLoss
model_dict = {
    # TODO: Fill these networks
    'ce' : SupCEResNet,
    'supcon' : SupAUCMResNet,
    'AUCM' : SupAUCMResNet,
}

loss_dict = {
    # TODO: Fill these Lossses
    'ce' : torch.nn.CrossEntropyLoss,
    'supcon' : SupConLoss,
    'focal' : FocalLoss,
    'AUCM' : AUCMLoss,
}
def set_model(opt):
    model = model_dict[opt.loss](name=opt.model, num_classes=opt.num_classes)
    if opt.loss == 'ce':
        criterion = loss_dict[opt.loss]
    elif opt.loss == 'supcon':
        criterion = loss_dict[opt.loss](temperature=opt.temperature)
    elif opt.loss == 'focal':
        criterion = loss_dict[opt.loss](gamma=opt.gamma)
    elif opt.loss == 'AUCM':
        criterion = loss_dict[opt.loss](temperature=opt.temperature)
    else:
        raise ValueError('Loss not supported: {}'.format(opt.loss))
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    
    return model, criterion    
    