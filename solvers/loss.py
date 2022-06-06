import torch
import torch.nn as nn
import logging

from .mmce import MMCE_weighted
from .flsd import FocalLossAdaptive
import sys
sys.path.append("..")
from utils import crl_utils

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()
# from https://openreview.net/pdf?id=NJS8kp15zzH
class InverseFocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(InverseFocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1+pt)**self.gamma * logpt
        
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class CRL(nn.Module):
    def __init__(self,arguments,history, **kwargs) -> None:
        super(CRL, self).__init__()
        self.args=arguments
        self.history = history
        self.criterion = nn.MarginRankingLoss(margin=0.0).cuda()

    def forward(self, logits, targets, idx):
        if self.args.rank_target == 'softmax':
            conf = nn.functional.softmax(logits, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif self.args.rank_target == 'entropy':
            if self.args.dataset == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = crl_utils.negative_entropy(logits,
                                                    normalize=True,
                                                    max_value=value_for_normalizing)
        # margin
        elif self.args.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(logits), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]

        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = self.history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        return self.criterion(rank_input1, rank_input2, rank_target)

class ClassficationAndCRL(nn.Module):
    def __init__(self, arguments,history,**kwargs):
        super(ClassficationAndCRL, self).__init__()
        self.history = history
        self.args = arguments
        self.classification_loss = nn.CrossEntropyLoss()
        self.CRL = CRL(arguments,history)

    def forward(self, logits, targets,idx):
        loss_cls = self.classification_loss(logits, targets)
        loss_ref = self.CRL(logits, targets,idx)
        return loss_cls + self.args.gamma * loss_ref

class ClassficationAndCRLAndMDCA(nn.Module):
    def __init__(self, arguments,history,**kwargs):
        super(ClassficationAndCRLAndMDCA, self).__init__()
        self.history = history
        self.args = arguments
        self.ClassficationAndCRL = ClassficationAndCRL(arguments,history)
        self.MDCA = MDCA()

    def forward(self, logits, targets,idx):
        loss_cal = self.MDCA(logits, targets)
        loss_ref = self.ClassficationAndCRL(logits, targets,idx)
        return loss_ref + self.args.beta * loss_cal

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss


class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="NLL+CRL", alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
        elif "CRL" in loss:
            self.classification_loss = CRL(gamma=self.gamma)
        else:
            self.classification_loss = LabelSmoothingLoss(alpha=self.alpha) 
        self.MDCA = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return loss_cls + self.beta * loss_cal

class BrierScore(nn.Module):
    def __init__(self, **kwargs):
        super(BrierScore, self).__init__()

    def forward(self, logits, target):
        
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class DCA(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        output = torch.softmax(logits, dim=1)
        conf, pred_labels = torch.max(output, dim = 1)
        calib_loss = torch.abs(conf.mean() -  (pred_labels == targets).float().mean())
        return self.cls_loss(logits, targets) + self.beta * calib_loss

class MMCE(nn.Module):
    def __init__(self, beta=2.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mmce = MMCE_weighted()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        cls = self.cls_loss(logits, targets)
        calib = self.mmce(logits, targets)
        return cls + self.beta * calib

class FLSD(nn.Module):
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)


loss_dict = {
    "focal_loss" : FocalLoss,
    "cross_entropy" : CrossEntropy,
    "LS" : LabelSmoothingLoss,
    "NLL+MDCA" : ClassficationAndMDCA,
    "LS+MDCA" : ClassficationAndMDCA,
    "FL+MDCA" : ClassficationAndMDCA,
    "brier_loss" : BrierScore,
    "NLL+DCA" : DCA,
    "MMCE" : MMCE,
    "FLSD" : FLSD,
    "IFL" : InverseFocalLoss,
    "CRL" : ClassficationAndCRL,
    "CRL+MDCA" : ClassficationAndCRLAndMDCA
}