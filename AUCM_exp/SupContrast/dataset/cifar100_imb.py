import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import _pickle as pk
from collections import Counter
from sklearn.utils import shuffle
import math
import pandas as pd
from abc import abstractmethod


class BaseDataset(Dataset):
    """
    torch.utils.data.Dataset
    """
    def __init__(self, x, y, transform=None):
        super(BaseDataset, self).__init__()
        self.transform = transform
        self.feat_dim = x.shape[1]
        self.class_dim = len(set(y))
        self.x, self.y = self.resample(x, y)
        self.cls_num_list = pd.Series(
            self.y).value_counts().sort_index().values
        self.freq_info = [
            num * 1.0 / sum(self.cls_num_list) for num in self.cls_num_list
        ]
        self.num_classes = len(self.cls_num_list)

    @abstractmethod
    def resample(self, x, y):
        pass

    def get_feat_dim(self):
        return self.feat_dim

    def get_class_dim(self):
        return self.class_dim

    def get_cls_num_list(self):
        return self.cls_num_list

    def get_freq_info(self):
        return self.freq_info

    def get_num_classes(self):
        return self.num_classes

    def get_labels(self):
        return self.y

    def __getitem__(self, index):
        # print(self.x[index])
        if self.transform is not None:
            return self.transform(self.x[index]), self.y[index]
        return np.squeeze(self.x[index]), self.y[index]

    def __len__(self):
        return len(self.y)


class VanillaDataset(BaseDataset):
    """
    Dataset without any resampling
    """
    def resample(self, x, y):
        return x, y


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        self.class_vector = class_vector
        # self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size

        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)
        y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})
        self.class_batch_size = {
            k: math.ceil(n * batch_size / y.shape[0])
            for k, n in y_counter.items()
        }
        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n)

        # sampling for each batch
        data = self.data.copy()
        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(
                    batch) == self.real_batch_size, 'not enough instances!'
            except (ValueError, AssertionError):
                break
            # print('sampled a batch ...')
            result.extend(shuffle(batch.index))
            data.drop(index=batch.index, inplace=True)
        return result

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def get_datasets(data_path, resampler_type,transform=None):
    # read data from disk
    # train data 
    with open(data_path + 'imb_cifar_train.pkl', 'rb') as f:
        X_train = pk.load(f,encoding="utf8")
        y_train = pk.load(f,encoding="utf8")
        # print(X_train)
        # print(y_train)
    # val data
    with open(data_path + 'imb_cifar_val.pkl', 'rb') as f:
        X_val = pk.load(f,encoding="utf8")
        y_val = pk.load(f,encoding="utf8")
    # test data 
    with open(data_path + 'imb_cifar_test.pkl', 'rb') as f:
        X_test = pk.load(f,encoding="utf8")
        y_test = pk.load(f,encoding="utf8")
    # print(X_train)
    # X_train type
    # print(type(X_train))
    # construct training set
    
    train_set = VanillaDataset(X_train, y_train,transform=transform)
    
    # construct val/test set
    val_set = VanillaDataset(X_val, y_val,transform=transform)
    test_set = VanillaDataset(X_test, y_test,transform=transform)
    return train_set, val_set, test_set


def get_data_loaders(data_path,
                     train_batch_size,
                     test_batch_size,transform=None):
    train_set, val_set, test_set = get_datasets(data_path,None,transform)
    sampler = StratifiedSampler(train_set.get_labels(),
                                train_batch_size)
    train_loader = DataLoader(train_set,
                              batch_size=sampler.real_batch_size,
                              shuffle=False,
                              sampler=sampler)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False)
    return train_loader, val_loader, test_loader

