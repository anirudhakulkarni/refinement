from lib2to3.pgen2.token import BACKQUOTE
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss, AUCM_MultiLabel_MDCA
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert
from libauc.metrics import auc_roc_score  # for multi-task
from calibration_library.metrics import ECELoss, SCELoss

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.nn as nn

# paramaters
SEED = 123
BATCH_SIZE = 256
epochs = 35
lr = 0.1
gamma = 500
weight_decay = 1e-5
margin = 1.0

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# dataloader
root = '../../../MDCA-Calibration-main/data/CheXpert-v1.0-small/'
# Index: -1 denotes multi-label mode including 5 diseases
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False,
                     use_frontal=True, image_size=224, mode='train', class_index=-1, verbose=False)
testSet = CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False,
                   use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False)
trainloader = torch.utils.data.DataLoader(
    traindSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

# check imbalance ratio for each task
print(traindSet.imratio_list)


# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=True, last_activation=None,
                    activations='relu', num_classes=5)
model = model.cuda()
model = nn.DataParallel(model)

# define loss & optimizer
Loss = AUCM_MultiLabel(num_classes=5)
optimizer = PESG(model,
                 a=Loss.a,
                 b=Loss.b,
                 alpha=Loss.alpha,
                 lr=lr,
                 gamma=gamma,
                 margin=margin,
                 weight_decay=weight_decay, device='cuda')

# training
best_val_auc = 0
for epoch in range(epochs):

    if (epoch+1) %10==0:
        optimizer.update_regularizer(decay_factor=2)

    for idx, data in enumerate(trainloader):
        train_data, train_labels = data
        train_data, train_labels = train_data.cuda(), train_labels.cuda()
        
        print(train_data.shape)
        y_pred = model(train_data)
        y_pred = torch.sigmoid(y_pred)
        loss = Loss(y_pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        if idx % 400 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = []
                test_true = []
                for jdx, data in enumerate(testloader):
                    test_data, test_labels = data
                    test_data = test_data.cuda()
                    y_pred = model(test_data)
                    y_pred = torch.sigmoid(y_pred)
                    test_pred.append(y_pred.cpu().detach().numpy())
                    test_true.append(test_labels.numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                val_auc_mean = roc_auc_score(test_true, test_pred)
                model.train()

                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    torch.save(model.state_dict(),
                               'aucm_multi_label_pretrained_model.pth')
                eces = ECELoss().loss(test_pred, test_true, n_bins=15)
                cces = SCELoss().loss(test_pred, test_true, n_bins=15)
                print('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f, ECE=%.4f, SCE=%.4f' %
                        (epoch, idx, val_auc_mean, best_val_auc, eces, cces))
if True:
    # load the saved model
    model=DenseNet121(last_activation=None,
                    activations='relu', num_classes=5)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('aucm_multi_label_pretrained_model.pth'))
    # model = torch.load('aucm_multi_label_pretrained_model.pth')
    # model = model.cuda()
    # test the model
    model.cuda()
    model.eval()
    testSet=CheXpert(csv_path='../MDCA-Calibration-main/data/CheXpert-v1.0-small/valid.csv',image_root_path='../MDCA-Calibration-main/data/CheXpert-v1.0-small/',use_upsampling=False,use_frontal=True,image_size=224,mode='valid',class_index=-1,verbose=False)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    # predict
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    # calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    print('AUC: ', auc)
    # calculate ECE
    ece = ECELoss().loss(y_pred, y_true, n_bins=15)
    print('ECE: ', ece)
    # calculate SCE
    sce = SCELoss().loss(y_pred, y_true, n_bins=15)
    print('SCE: ', sce)
    # calculate accuracy
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    acc = np.mean(y_pred == y_true)
    print('Accuracy: ', acc)

# eces = ECELoss().loss(test_pred, test_true, n_bins=15)
# cces = SCELoss().loss(test_pred, test_true, n_bins=15)
# aucscore = auc_roc_score(test_true, test_pred)
# print("ECE: ", eces)
# print("SCE: ", cces)
# print("AUC: ", aucscore)

# accuracies = np.equal(test_pred,test_true)
# print("Accuracies: ", np.count_nonzero(accuracies)/accuracies.size)
# print("Mean Accuracies: ", np.mean(np.count_nonzero(accuracies,axis=1)/accuracies.shape[1]))
