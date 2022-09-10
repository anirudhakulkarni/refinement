import torch 
import torch.nn.functional as F


class CrossEntropyLoss(torch.nn.Module):
    """
    Cross Entropy Loss with Sigmoid Function
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)
    
class FocalLoss(torch.nn.Module):
    """
    Focal Loss
    Reference: 
        https://amaarora.github.io/2020/06/29/FocalLoss.html
    """
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
