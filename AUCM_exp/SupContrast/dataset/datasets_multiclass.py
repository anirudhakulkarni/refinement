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
from torchvision.datasets import CIFAR100, iNaturalist, ImageNet
import torch

class TraditionalDataset(Dataset):
    def __init__(self, data_name, mode):
        self.data_name = data_name
        self.mode = mode
        self.base_path='data_index/'+data_name+'.npz/'
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







def get_train_test_val_loader(opt):
    data_name = opt.dataset
    if data_name == 'cifar100':
        train_dataset = CIFAR100(root=data_folder, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = CIFAR100(root=data_folder, train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        return train_loader, test_loader
    elif data_name == 'iNaturalist':
        train_dataset = iNaturalist(root=data_folder, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = iNaturalist(root=data_folder, train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        return train_loader, test_loader
            
    elif data_name in datas:
        train_dataset = TraditionalDataset(data_name, 'train')
        test_dataset = TraditionalDataset(data_name, 'test')
        val_dataset = TraditionalDataset(data_name, 'val')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        return train_loader, test_loader
                        # , val_loader    
    
    elif data_name == 'imagenet':
        train_dataset = ImageNet(root= opt.data_folder, split='train', download=True, transform=transforms.ToTensor())
        test_dataset = ImageNet(root= opt.data_folder, split='val', download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        return train_loader, test_loader
        
    else:
        raise ValueError('dataset not found')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
