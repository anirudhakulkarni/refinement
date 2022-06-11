from numpy import argsort
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

# https://github.com/pytorch/pytorch/blob/cde0cefa1cbd7fac553880f5e77a99d771bf837a/torch/_refs/nn/functional/__init__.py#L258
class MarginRankingLossPow(nn.Module):
    def __init__(self, margin: float = 0., p=1) -> None:
        super(MarginRankingLossPow, self).__init__( )
        self.margin = margin
        self.p=p

    def forward(self,input1,input2,target):
        # loss_without_reduction = max(0, −target * (input1 − input2)^2 + margin)
        neg_target = -target
        input_diff = torch.pow(input1-input2,self.p)-1
        mul_target_input = neg_target*input_diff
        add_margin = mul_target_input+self.margin
        zeros=torch.zeros_like(add_margin)
        loss = torch.max(add_margin, zeros)
        return loss.mean()


class MarginRankingLossScale(nn.Module):
    def __init__(self, p=1.25) -> None:
        super(MarginRankingLossScale, self).__init__( )
        self.p=p

    def forward(self,input1,input2,target):
        # loss_without_reduction = max(0, −target * p*(input1 − input2) + margin)
        neg_target = -target
        input_diff = (input1-input2)*self.p
        mul_target_input = neg_target*input_diff
        add_margin = mul_target_input
        zeros=torch.zeros_like(add_margin)
        loss = torch.max(add_margin, zeros)
        return loss.mean()


class MarginRankingLossSmooth(nn.Module):
    def __init__(self) -> None:
        super(MarginRankingLossSmooth, self).__init__( )

    def forward(self,input1,input2,target,margin):
        '''
        # loss_without_reduction =  0 if target * (input1 − input2) >= margin
                                    1/(2*margin) * (margin - target * (input1 − input2))^2 elsif 0 < target * (input1 − input2) < margin
                                    margin/2 - target * (input1 − input2)^2 else
        '''
        epsilon=1e-12
        margin=margin+epsilon
        neg_target = -target
        input_diff = input1-input2
        mul_target_input = neg_target*input_diff
        add_margin = mul_target_input+margin
        interval1_mask = (mul_target_input>=margin).float()
        interval2_mask = (mul_target_input<margin).float()
        interval3_mask = (mul_target_input<=0).float()
        # interval2_mask is interscetion of interval2_mask and negation of interval3_mask
        interval2_mask = interval2_mask*(1-interval3_mask)
        loss = interval1_mask*0+interval2_mask*0.5*torch.pow(add_margin,2)/margin+interval3_mask*(margin/2-mul_target_input)
        return loss.mean()
    

class MarginRankingLossExp(nn.Module):
    def __init__(self) -> None:
        super(MarginRankingLossExp, self).__init__( )

    def forward(self,input1,input2,target):
        # loss_without_reduction = max(0, −target * e^(input1 − input2) + margin)
        neg_target = -target
        input_diff = torch.exp(input2-input1)
        mul_target_input = neg_target*input_diff
        add_margin = mul_target_input
        zeros=torch.zeros_like(add_margin)
        loss = torch.max(add_margin, zeros)
        return loss.mean()



class CRL(nn.Module):
    def __init__(self,arguments,history, **kwargs) -> None:
        super(CRL, self).__init__()
        self.args=arguments
        self.history = history
        # self.criterion = nn.MarginRankingLoss(margin=0.0).cuda()
        # self.criterion = MarginRankingLossSq(margin=0.0,p=3).cuda()
        if "exp" in arguments.loss:
            self.criterion=MarginRankingLossExp().cuda()
        elif "cubic" in arguments.loss:
            self.criterion=MarginRankingLossPow(margin=0.0,p=3).cuda()
        elif "square" in arguments.loss:
            self.criterion=MarginRankingLossPow(margin=0.0,p=2).cuda()
        elif "scale" in arguments.loss:
            self.criterion=MarginRankingLossScale(p=1.25).cuda()
        elif "smooth" in arguments.loss:
            self.criterion=MarginRankingLossSmooth().cuda()
        else:
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
        
        idx1 = idx
        idx2 = torch.roll(idx, -1)

        # rank_target is the indicator function with 1 if x1>x2 or -1 if x1<x2 or 0 if x1=x2 on correctness
        # margin is absolute difference in correctness. |x1-x2|

        rank_target, rank_margin = self.history.get_target_margin(idx1, idx2)
        if "smooth" in self.args.loss:
            return self.criterion(rank_input1, rank_input2, rank_target, rank_margin) 
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
        # rank_input2 is confidence of 2nd pair + absolute value of confidence difference
        return self.criterion(rank_input1, rank_input2, rank_target)

class ClassficationAndCRL(nn.Module):
    def __init__(self,loss, arguments,history,**kwargs):
        super(ClassficationAndCRL, self).__init__()
        self.history = history
        self.args = arguments
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        # elif "FL" in loss:
        else:
            self.classification_loss = FocalLoss(gamma=self.args.gamma)
        self.CRL = CRL(arguments,history)
        self.recordrefinementloss=0.0
        self.recordclassification=0.0
        self.counter=0

    def forward(self, logits, targets,idx):
        loss_cls = self.classification_loss(logits, targets)
        loss_ref = self.CRL(logits, targets,idx)
        self.counter+=1
        self.recordclassification+=loss_cls
        self.recordrefinementloss+=loss_ref
        if self.counter>=704:
            print("focal loss:",self.recordclassification/self.counter, "refinement loss:",self.recordrefinementloss/self.counter)
            logging.info("class loss={}, refinement loss={}".format(self.recordclassification/self.counter, self.recordrefinementloss/self.counter))
            self.counter=0
            self.recordrefinementloss=0.0
            self.recordclassification=0.0
                
        return loss_cls + self.args.theta * loss_ref

class ClassficationAndCRLAndMDCA(nn.Module):
    def __init__(self, loss,arguments,history,**kwargs):
        super(ClassficationAndCRLAndMDCA, self).__init__()
        self.history = history
        self.args = arguments
        self.ClassficationAndCRL = ClassficationAndCRL(loss,arguments,history)
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
    "NLL+CRL" : ClassficationAndCRL,
    "NLL+CRLexp" : ClassficationAndCRL,
    "NLL+CRLcubic" : ClassficationAndCRL,
    "NLL+CRLsquare" : ClassficationAndCRL,
    "NLL+CRL+MDCA" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCA" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCAexp" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCAcubic" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCAsquare" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCAscale" : ClassficationAndCRLAndMDCA,
    "FL+CRL+MDCAsmooth" : ClassficationAndCRLAndMDCA
    
}