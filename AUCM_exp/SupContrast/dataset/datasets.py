from PIL import Image
from torch.utils.data import Dataset
from libauc.utils import ImbalancedDataGenerator #BUG:  this is using import from conda install
import numpy as np
# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from libauc.datasets import CAT_VS_DOG, CIFAR10, CIFAR100, STL10, Melanoma
from utils.util import TwoCropTransform, NoiseTransform
from .cifar_lt import IMBALANCECIFAR100, IMBALANCECIFAR10
from .imagenet_lt import set_loader as set_loader_imagenet_lt
from .cifar100_imb import get_data_loaders as set_loader_cifar100_imb

def make_deterministic(SEED=123):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED=123
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




class ImageDataset(Dataset):
    def __init__(self, images, targets,transform_train, transform_val, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transform_train
        self.transform_val = transform_val

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        elif self.mode == 'val':
            image = self.transform_val(image)
        return image, target



def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar10_lt' or opt.dataset == 'cifar100_imb':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'cifar100_lt':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'c2':
        mean = (0.33554432, 0.33554432, 0.33554432)
        std = (0.28430098, 0.2612929,  0.24912025)
    elif opt.dataset == 'stl10':
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2564, 0.2762)
    elif opt.dataset == 'melanoma':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'imagenet_lt':
        return set_loader_imagenet_lt(opt)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
        # pass
    if opt.dataset == 'cifar100_imb':
        
        if opt.loss == 'supcon':
            train_transform = transforms.Compose([
            ])
            train_transform = TwoCropTransform(train_transform)
            train_loader, val_loader, _ = set_loader_cifar100_imb(opt.data_folder+'/cifar-100-imb/', opt.batch_size, opt.batch_size, transform=train_transform)
        else:            
            train_loader, val_loader, _ = set_loader_cifar100_imb(opt.data_folder+'/cifar-100-imb/', opt.batch_size, opt.batch_size)
        return train_loader, val_loader
    
       
    
    normalize = transforms.Normalize(mean=mean, std=std)
    print("Using delta = ",opt.delta)
    normalize = NoiseTransform(normalize,delta=opt.delta)

    if opt.loss!='supcon':
        # TODO: supcon name is different
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        # only for supcon
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])    
    if opt.dataset == 'cifar10':
        train_data, train_targets = CIFAR10(root=opt.data_folder)
        val_data, val_targets = CIFAR10(root=opt.data_folder, train=False)
    elif opt.dataset == 'cifar100':
        train_data, train_targets = CIFAR100(root=opt.data_folder)
        val_data, val_targets = CIFAR100(root=opt.data_folder, train=False)
    elif opt.dataset == 'c2':
        train_data, train_targets = CAT_VS_DOG(root=opt.data_folder)
        val_data, val_targets = CAT_VS_DOG(root=opt.data_folder, train=False)
    elif opt.dataset == 'stl10':
        train_data, train_targets = STL10(root=opt.data_folder)
        val_data, val_targets = STL10(root=opt.data_folder, split='test')
        train_data = train_data.transpose(0, 2, 3, 1)
        val_data = val_data.transpose(0, 2, 3, 1)
    elif opt.dataset == 'melanoma':
        train_set = Melanoma(root=opt.data_folder+'/melanoma/', is_test=False, test_size=0.2)
        test_set = Melanoma(root=opt.data_folder+'/melanoma/', is_test=True, test_size=0.2)
        train_set.transforms = TwoCropTransform(train_set.transforms) if opt.loss=='supcon' else train_set.transforms
        test_set.transforms = TwoCropTransform(test_set.transforms) if opt.loss=='supcon' else test_set.transforms
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        return train_loader, test_loader
    elif opt.dataset == 'cifar10_lt':
        # https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/cifar_train.py#L153
        
        train_dataset = IMBALANCECIFAR10(root= opt.data_folder, imb_factor=args.im_ratio, rand_number=SEED, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, download=True, transform=val_transform)
        train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True)
        return train_loader, val_loader
    elif opt.dataset == 'cifar100_lt':
        # https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/cifar_train.py#L153
        train_dataset = IMBALANCECIFAR100(root= opt.data_folder, imb_factor=opt.im_ratio, rand_number=SEED, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, download=True, transform=val_transform)
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True)
        return train_loader, val_loader
            
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    
    train_transform = TwoCropTransform(train_transform) if opt.loss=='supcon' else train_transform
    # val_transform = TwoCropTransform(val_transform) if opt.loss=='supcon' else val_transform

    generator = ImbalancedDataGenerator(verbose=True, random_seed=SEED)
    (train_images, train_labels) = generator.transform(
        train_data, train_targets, imratio=opt.imratio)
    train_loader = torch.utils.data.DataLoader(ImageDataset(
        train_images, train_labels,train_transform,val_transform,mode='train'), 
        batch_size=opt.batch_size, shuffle=(train_sampler is None), num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    (val_images, val_labels) = generator.transform(
        val_data, val_targets, imratio=0.5) #NOTE: Default testing is at 0.5
    val_loader = torch.utils.data.DataLoader(ImageDataset(
        val_images, val_labels,train_transform,val_transform,mode='val'), 
        batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    del train_data, train_targets, val_data, val_targets, train_images, train_labels, val_images, val_labels
    return train_loader, val_loader

