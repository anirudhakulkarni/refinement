from tkinter import image_names
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
import sys
# paramaters
SEED = 123
BATCH_SIZE = 256
lr = 0.1
gamma = 500
weight_decay = 1e-5
margin = 1.0
epochs = 35


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>
inputcsv = sys.argv[1]
outputcsv = sys.argv[2]

if True:
    # load the saved model
    model = DenseNet121(last_activation=None,
                        activations='relu', num_classes=5)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(
        'aucm_multi_label_MDCA_pretrained_model.pth'))
    # model = torch.load('aucm_multi_label_pretrained_model.pth')
    # model = model.cuda()
    # test the model
    model.cuda()
    model.eval()

    testSet = CheXpert(csv_path=inputcsv, image_root_path='../data/CheXpert-v1.0-small/',
                       use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False,shuffle=False)
    # print the test set size
    print(len(testSet))
    print(testSet._images_list)
    print(len(testSet._images_list))
    testloader = torch.utils.data.DataLoader(
        testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    # predict
    # print(testloader)
    # print number of images in the test set
    print(len(testloader.dataset))
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
    # y_pred = np.argmax(y_pred, axis=1)
    # y_true = np.argmax(y_true, axis=1)
    # acc = np.mean(y_pred == y_true)
    # print('Accuracy: ', acc)
    # save predictions
    # concatenate image names and predictions
    image_names=np.loadtxt(inputcsv, dtype=str, delimiter=',', skiprows=1, usecols=0)
    # append image names as first column
    print(y_pred)
    print(image_names)
    print(y_pred.shape)
    print(image_names.shape)
    # concatenate testSet._images_list which is 1d array and y_pred
    y_pred = np.concatenate((np.array(testSet._images_list).reshape(-1,1), y_pred), axis=1)
    # y_pred = np.concatenate((testSet._images_list, y_pred), axis=1)
    # save predictions
    np.savetxt(outputcsv, y_pred, delimiter=',', fmt='%s')
    
# eces = ECELoss().loss(test_pred, test_true, n_bins=15)
# cces = SCELoss().loss(test_pred, test_true, n_bins=15)
# aucscore = auc_roc_score(test_true, test_pred)

# print("ECE: ", eces)
# print("SCE: ", cces)
# print("AUC: ", aucscore)

# if test_pred.shape != test_true.shape:
#     # change from one-hot to class index
#     labels = np.argmax(test_true, axis=1)
# # print(self.predictions)
# accuracies = np.equal(test_pred, test_true)
# # print("Accuracies: ", np.count_nonzero(accuracies,axis=0)/accuracies.size)
# print("Accuracy: ", np.mean(accuracies))
