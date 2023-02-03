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
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive Loss
        L_{c}=\sum_{x \in S} \sum_{x^{\prime} \in S}(1-y)\left(h_{w}(x)-h_{w}\left(x^{\prime}\right)\right)^{2}-y\left(h_{w}(x)-h_{w}\left(x^{\prime}\right)\right)^{2} \\
        where
        y = 0 if x and x' are in the same class
        y = 1 if x and x' are in different classes
        S = set of all training samples
        S+ = set of positive samples
        S- = set of negative samples
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        """
        y_pred is h_w(x)
        y_true is indicator of positive or negative pair
        iterate over all samples and use y=0 for same class and y=1 for different class
        """
        # print dimensions
        assert y_pred.shape == y_true.shape

        loss = torch.tensor(0.0).cuda()
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[0]):
                if i == j:
                    continue
                if y_true[i] == y_true[j]:
                    loss += (y_pred[i][0] - y_pred[j][0])**2
                else:
                    loss += - (y_pred[i][0] - y_pred[j][0])**2
                    # loss += max(0, self.margin - (y_pred[i] - y_pred[j])**2)
        return loss


class ContrastiveLossOptimized(torch.nn.Module):
    """
    Compute contrastive loss with matrix multiplication
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLossOptimized, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true, epoch):

        assert y_pred.shape == y_true.shape
        if  epoch<5:
            return F.binary_cross_entropy_with_logits(y_pred, y_true)
        # else:
        #     y_pred = torch.sigmoid(y_pred)
        # use matrix multiplcation
        y_pred = torch.sigmoid(y_pred)
        loss = torch.tensor(0.0).cuda()
        # instead of iterating over all samples, use matrix multiplication

        # compute all pairwise distances
        # create matrix from 1d vector
        # print(y_pred)
        # print(y_true)
        # s
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        # print("Ypred")
        # print(y_pred)
        # print(y_pred.repeat(y_pred.size(0), 1))
        # print(y_pred.repeat(y_pred.size(0), 1).T)
        # print(y_pred.repeat(y_pred.size(0), 1) - y_pred.repeat(y_pred.size(0), 1).T)
        dist = y_pred.repeat(y_pred.size(0), 1) - \
            y_pred.repeat(y_pred.size(0), 1).T
        # dist = torch.pow(y_pred.view(-1, 1)-y_pred.view(1, -1), 2)
        # dist = torch.clamp(dist, min=1e-12).sqrt()  # for numerical stability
        # print(dist.shape)
        # print(y_pred.repeat(y_pred.shape[0], 1).shape)
        # compute indicator matrix
        y_true = y_true.repeat(y_true.size(0), 1)
        # print(y_true)
        indicator = (y_true != y_true.t()).float()
        # print(indicator)
        # print(dist)
        # compute loss
        loss = (1 - indicator) * torch.pow(dist, 2) + \
            indicator * torch.pow(1-dist, 2)
        loss = loss.sum()
        print(loss)
        return loss



class ContrastiveLossOptimizedV1(torch.nn.Module):
    """
    Compute contrastive loss with matrix multiplication
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLossOptimizedV1, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true, epoch):
        # print(y_pred.shape, y_true.shape)
        assert y_pred.shape == y_true.shape
        # if  epoch<50:
            # write cross entropy loss from scratch
            # y_pred = torch.sigmoid(y_pred)
            # logloss = -y_true*torch.log(y_pred)-(1-y_true)*torch.log(1-y_pred)
            # return F.binary_cross_entropy_with_logits(y_pred, y_true)
            # return logloss.mean()
            # return loss
        # if  epoch<5:
        #     return F.binary_cross_entropy_with_logits(y_pred, y_true)
        # else:
        #     y_pred = torch.sigmoid(y_pred)
        # logits to be converted to probabilities
        y_pred = torch.sigmoid(y_pred)
        loss = torch.tensor(0.0).cuda()
        # convert to 1d vector: (batch_size, 1) -> (batch_size)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        # convert to 2d matrix: (batch_size) -> (batch_size, batch_size)
        dist = y_pred.repeat(y_pred.size(0), 1) - \
            y_pred.repeat(y_pred.size(0), 1).T
        # convert to 2d matrix: (batch_size) -> (batch_size, batch_size)
        y_true_2d = y_true.repeat(y_true.size(0), 1)
        # compute indicator matrix
        indicator = (y_true_2d != y_true_2d.t()).float()
        # different classes then loss = distance square
        loss_diff = (indicator) * torch.sum(torch.pow(dist, 2))
        # same classes then loss = variance of positive samples + variance of negative samples
        mean_pos = torch.mean(y_pred[y_true == 1])
        mean_neg = torch.mean(y_pred[y_true == 0])
        # print(mean_pos, mean_neg)
        loss_same = (1 - indicator) * (torch.sum(torch.pow(y_pred[y_true == 1] - mean_pos, 2)) + torch.sum(torch.pow(y_pred[y_true == 0] - mean_neg, 2)))
        loss = torch.mean(loss_diff) + torch.mean(loss_same)
        loss = loss.mean()
        # print(torch.sum(indicator),torch.sum(1-indicator))
        print(y_pred)
        print(y_true)
        print(loss_diff, loss_same, loss)
        return loss


class ContrastiveLossOptimizedV2(torch.nn.Module):
    """
    Compute contrastive loss with matrix multiplication
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLossOptimizedV2, self).__init__()
        self.margin = margin
    
    def forward(self, y_pred, y_true, epoch,model):
        y_pred = torch.sigmoid(y_pred)
        print(torch.sum(y_pred), torch.sum(y_true))
        loss = torch.tensor(0.0).cuda()
        mean_pos = torch.mean(y_pred[y_true == 1])
        mean_neg = torch.mean(y_pred[y_true == 0])
        loss = loss + torch.sum(torch.pow(y_pred[y_true == 1] - mean_pos, 2)) + torch.sum(torch.pow(y_pred[y_true == 0] - mean_neg, 2))
        # print(loss)
        # add all pair distance between positive and negative samples
        # loss = loss + torch.sum(torch.pow(y_pred[y_true == 1].repeat(y_pred[y_true == 0].size(0), 1) - y_pred[y_true == 0].repeat(y_pred[y_true == 1].size(0), 1).T, 2))
        loss = loss + (1-mean_pos+mean_neg)**2
        l2_reg = torch.tensor(0.0).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)
        # loss = loss + l2_reg
        print(mean_neg,mean_pos,(1-mean_pos+mean_neg)**2,loss)
        print(loss.mean())
        return loss.mean()
        loss = torch.tensor(0.0).cuda()
        mean_p