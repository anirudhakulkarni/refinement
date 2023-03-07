import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_cifar import ResNet20
from .densenet import DenseNet121
from .resnet_big import resnet18, resnet34, resnet50, resnet101
model_dict = {
    "resnet20": [ResNet20, 64],
    "resnet50": [resnet50, 2048],
    "densenet121": [DenseNet121, 1024],
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
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
        # self.sigmoid = torch.nn.LogSoftmax(dim=1)
        # self.sigmoid = torch.nn.Sigmoid()
        # TODO: Should we use sigmoid or softmax?

    def forward(self, x):
        # TODO: softmax needs to be fed before loss
        # features, num_classes

        # print(self.sigmoid(self.fc(self.encoder(x))))
        # return self.sigmoid(self.fc(self.encoder(x)))
        return self.fc(self.encoder(x))


class SupAUCMResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(SupAUCMResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        # TODO: Sigmoid transferred to output. model outputs logits
        # self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.Softmax()
        # TODO: Should we use sigmoid or softmax?

    def forward(self, x):
        # TODO: softmax needs to be fed before loss
        # return self.sigmoid(self.fc(self.encoder(x)))
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class Cifar100IMBModel(nn.Module):
    def __init__(self, name, input_dim=2048, num_classes=100, bias=True):
        super(Cifar100IMBModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 1024, bias=bias),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes),
            # torch.nn.Linear(64, 12),
            # torch.nn.Softmax(dim=1)
            ).cuda()

    def forward(self, x):
        x = x.float()
        return self.model(x)

class Cifar100IMBModelSupcon(nn.Module):
    def __init__(self, name, input_dim=2048, outdim=128, bias=True):
        super(Cifar100IMBModelSupcon, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 1024, bias=bias),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, outdim),
            # torch.nn.Linear(64, 12),
            # torch.nn.Softmax(dim=1)
            ).cuda()

    def forward(self, x):
        x = x.float()
        return self.model(x)

