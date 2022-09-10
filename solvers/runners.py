import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy, get_all_metrics, crl_accuracy
import numpy as np
from calibration_library.metrics import ECELoss, SCELoss
from sklearn.metrics import roc_auc_score
from utils import parse_args
args=parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def train(trainloader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    # print("bar",len(trainloader))
    for batch_idx, (inputs, targets) in bar:
        # print(inputs)
        # print(inputs.shape)
        # print(targets)
        # print(targets.shape)
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg
                    ))

    return (losses.avg, top1.avg)

# epoch wise loss correctness calculating training procedure. Refer to CRL
def train_CRL(trainloader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    # print("bar",len(trainloader))
    # for batch_idx, (inputs, targets) in bar:
    for batch_idx, (inputs, targets, idx) in bar:
        
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets, idx)

        # measure accuracy and record loss
        prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg
                    ))
        _,correct=crl_accuracy(outputs, targets)
        criterion.history.correctness_update(idx, correct, outputs)
    criterion.history.max_correctness_update()


    return (losses.avg, top1.avg)

@torch.no_grad()
def test(testloader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    auroc_dict = get_all_metrics(all_targets,all_outputs)
    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, auroc_dict)

@torch.no_grad()
def get_logits_from_model_dataloader(testloader, model):
    """Returns torch tensor of logits and targets on cpu"""
    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(testloader, total=len(testloader), desc="Evaluating logits")
    for inputs, targets in bar:
        inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        # to numpy
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

    
@torch.no_grad()
def test_CRL(testloader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets,idx) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        # print(criterion)
        if "CRL" in str(criterion):
            loss = criterion(outputs, targets, idx)
        else:
            loss = criterion(outputs, targets)
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    auroc_dict = get_all_metrics(all_targets,all_outputs)
    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, auroc_dict)
