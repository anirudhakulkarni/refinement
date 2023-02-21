from networks.main import SupCEResNet, SupAUCMResNet, SupConResNet, LinearClassifier
import torch
from .losses import SupConLoss, FocalLoss, AUCMLoss, ClassficationAndMDCA
import torch.backends.cudnn as cudnn
model_dict = {
    # TODO: Fill these networks
    'ce' : SupCEResNet,
    'focal' : SupCEResNet,
    'supcon' : SupConResNet,
    'aucm' : SupAUCMResNet,
    'aucs' : SupAUCMResNet,
    'ce_linear': LinearClassifier,
    'mdca_linear': LinearClassifier,
}

loss_dict = {
    # TODO: Fill these Lossses
    'ce' : torch.nn.CrossEntropyLoss,
    'supcon' : SupConLoss,
    'focal' : FocalLoss,
    'aucm' : AUCMLoss,
    'aucs' : AUCMLoss,
    'ce_linear': torch.nn.CrossEntropyLoss,
    'mdca_linear': ClassficationAndMDCA
}
def set_model(opt):
    if opt.loss == 'supcon':
        model = model_dict[opt.loss](name=opt.model)
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

