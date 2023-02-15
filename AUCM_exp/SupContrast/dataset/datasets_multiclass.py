'''
create dataloader for datasets from "Learning with multiclass AUC" paper
Data is stored in npy files
Convert them to torch dataloader
1. Traditional datasets: balance, dermatology, ecoli, new-thyroid, pageblocks, segmentImb, shuttle, svmguide2, yeast
2. Other datasets: cifar100, iNaturalist, User-Imb
'''


datas=['balance', 'dermatology', 'ecoli', 'new-thyroid', 'pageblocks', 'segmentImb', 'shuttle', 'svmguide2', 'yeast']

# dummy example
data_name='balance'
base_path='data_index/'+data_name+'.npz/'
tot_X=base_path+'tot_X.npy'
tot_Y=base_path+'tot_Y.npy'
train_ids=base_path+'shuffle_train.npy'
test_ids=base_path+'shuffle_test.npy'
val_ids=base_path+'shuffle_val.npy'

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100, INaturalist, ImageFolder
from imagenet import ImageNetKaggle
import torch
import os
class TraditionalDataset(Dataset):
    def __init__(self, data_name, mode):
        self.data_name = data_name
        self.mode = mode
        self.base_path='data_multiclass/data_index/'+data_name+'.npz/'
        self.tot_X=self.base_path+'tot_X.npy'
        self.tot_Y=self.base_path+'tot_Y.npy'
        self.train_ids=self.base_path+'shuffle_train.npy'
        self.test_ids=self.base_path+'shuffle_test.npy'
        self.val_ids=self.base_path+'shuffle_val.npy'
        self.X = np.load(self.tot_X)
        self.Y = np.load(self.tot_Y)
        if self.mode == 'train':
            self.ids = np.load(self.train_ids)
        elif self.mode == 'test':
            self.ids = np.load(self.test_ids)
        elif self.mode == 'val':
            self.ids = np.load(self.val_ids)
        else:
            raise ValueError('mode should be train, test, or val')
        self.X = self.X[self.ids]
        self.Y = self.Y[self.ids]
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.int64)
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.transform(self.X[index]), self.Y[index]







def set_loader(opt):
    data_name = opt.dataset
    if data_name == 'cifar100':
        train_dataset = CIFAR100(root=opt.data_folder, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = CIFAR100(root=opt.data_folder, train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        return train_loader, test_loader
    elif data_name == 'iNaturalist':
        train_dataset = INaturalist(root=opt.data_folder, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = INaturalist(root=opt.data_folder, train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        return train_loader, test_loader
            
    elif data_name in datas:
        train_dataset = TraditionalDataset(data_name, 'train')
        test_dataset = TraditionalDataset(data_name, 'test')
        val_dataset = TraditionalDataset(data_name, 'val')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        return train_loader, test_loader
                        # , val_loader    
    
    elif data_name == 'imagenet':
        traindir = os.path.join(opt.data_folder, 'train')
        valdir = os.path.join(opt.data_folder, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))            
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)
        
        return train_loader, val_loader
    else:
        raise ValueError('dataset not found')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
