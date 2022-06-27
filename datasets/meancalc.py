import random

from torchvision import datasets
from torchvision import transforms
from torch.utils import data
import torch
# train_loader = data.DataLoader(train_set)
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
train_dataset = datasets.CIFAR100(root='../data', train=True, 
                                 transform=transforms.ToTensor(), download=True)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=64)
print(get_mean_and_std(train_dataloader))
train_dataset = datasets.CIFAR10(root='../data', train=True, 
                                 transform=transforms.ToTensor(), download=True)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=64)
print(get_mean_and_std(train_dataloader))
train_dataset = datasets.SVHN(root='../data', split="train", 
                                 transform=transforms.ToTensor(), download=True)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=64)
print(get_mean_and_std(train_dataloader))