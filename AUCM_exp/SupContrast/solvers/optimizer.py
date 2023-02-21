import torch.optim as optim
from libauc.optimizers import PESG


def set_optimizer(opt, model, criterion):

    # AUCM requires a different optimizer
    if opt.loss == 'aucm':
        optimizer = PESG(model,
                                a=criterion.a,
                                b=criterion.b,
                                alpha=criterion.alpha,
                                lr=opt.learning_rate,
                                gamma=opt.gamma,
                                margin=opt.margin,
                                weight_decay=opt.weight_decay)
        return optimizer

    else:
        # optimizer1 = optim.Adam(model.parameters(),
        #                     lr=opt.learning_rate,
        #                     weight_decay=opt.weight_decay)
        # optimizer1 = optim.Adam(model.parameters(),
        #                     lr=opt.learning_rate2,
        #                     weight_decay=opt.weight_decay)
        
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    # TODO: remove this to optimizer1, optimizer2
    # return optimizer1, optimizer1
    return optimizer