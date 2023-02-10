import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_cifar import ResNet20
from .densenet import DenseNet121
model_dict = {
    "resnet20":[ResNet20,64],
    "densenet121": [DenseNet121, 1024]
}

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        self.sigmoid = torch.nn.LogSoftmax()
        # TODO: Should we use sigmoid or softmax?

    def forward(self, x):
        # TODO: softmax needs to be fed before loss
        return self.sigmoid(self.fc(self.encoder(x)))
        # return self.fc(self.encoder(x))

class SupAUCMResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupAUCMResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        # TODO: Should we use sigmoid or softmax?

    def forward(self, x):
        # TODO: softmax needs to be fed before loss
        return self.sigmoid(self.fc(self.encoder(x)))
        # return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)
        # FIXME: WHERE IS SIGMOID?
        

    def forward(self, features):
        return self.fc(features)


