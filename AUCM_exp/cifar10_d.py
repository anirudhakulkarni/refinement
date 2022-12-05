from libauc.losses import AUCMLoss, AUCM_MDCALoss, CrossEntropyLoss
from libauc.optimizers import PESG
from libauc.models import densenet121
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from calibration_library.metrics import ECELoss, SCELoss
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import sys


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((crop_size, crop_size), padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target


# paramaters
SEED = 123
BATCH_SIZE = 128
imratio = 0.01  # for demo
lr = 0.1
weight_decay = 1e-4

gammas = [100, 300, 500, 700, 1000]
margins = [0.1, 0.3, 0.5, 0.7, 1.0]
# gammas = [500]
# margins = [ 1.0]

lossfunction = sys.argv[1]
if lossfunction=='ce':
    gammas = [300]
    margins = [ 1.0]

set_all_seeds(SEED)
# Tune hyperparameters, e.g., gammas and margins
for gamma in gammas:
    for margin in margins:

        # dataloader
        train_data, train_targets = CIFAR10(
            root='../../MDCA-Calibration-main/data', train=True)
        test_data, test_targets = CIFAR10(
            root='../../MDCA-Calibration-main/data', train=False)
        print("data found")
        generator = ImbalancedDataGenerator(verbose=True, random_seed=SEED)
        (train_images, train_labels) = generator.transform(
            train_data, train_targets, imratio=imratio)
        (test_images, test_labels) = generator.transform(
            test_data, test_targets, imratio=0.5)

        trainloader = torch.utils.data.DataLoader(ImageDataset(
            train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
        testloader = torch.utils.data.DataLoader(ImageDataset(
            test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)


        # You need to include sigmoid activation in the last layer for any customized models!
        model = densenet121(pretrained=False, last_activation=None, num_classes=1)
        model = model.cuda()
        if lossfunction == 'aucm':
            Loss = AUCMLoss()
        elif lossfunction == 'mdca':
            Loss = AUCM_MDCALoss()
        elif lossfunction == 'ce':
            Loss = CrossEntropyLoss()
        else:
            print("loss function not supported")
            exit()
        print("gamma: ", gamma)
        print("margin: ", margin)
        if lossfunction == 'aucm' or lossfunction == 'mdca':
            optimizer = PESG(model,
                             a=Loss.a,
                             b=Loss.b,
                             alpha=Loss.alpha,
                             lr=lr,
                             gamma=gamma,
                             margin=margin,
                             weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50,75], gamma=0.1)

        for epoch in range(100):

            train_pred = []
            train_true = []
            model.train()
            for data, targets in trainloader:
                data, targets = data.cuda(), targets.cuda()
                y_pred = model(data)
                # y_pred = torch.sigmoid(y_pred)
                loss = Loss(y_pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_pred.append(y_pred.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_auc = roc_auc_score(train_true, train_pred)

            model.eval()
            test_pred = []
            test_true = []
            for j, data in enumerate(testloader):
                test_data, test_targets = data
                test_data = test_data.cuda()
                y_pred = model(test_data)
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(test_targets.numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc = roc_auc_score(test_true, test_pred)
            model.train()
            scheduler.step()
            # eces = ECELoss().loss(test_pred, test_true, n_bins=15)
            # cces = SCELoss().loss(test_pred, test_true, n_bins=15)
            # print('Epoch: {} | Train_loss: {:4f} | Train AUC: {:.4f} | Val AUC: {:.4f} | ECE: {:.4f} | SCE: {:.4f}'.format(
            #     epoch, loss.item(), train_auc, val_auc, eces, cces))
            # print results
            print("epoch: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}, lr:{:4f}".format(
                epoch, loss.item(), train_auc, val_auc, optimizer.lr))

        # save results
        with open(lossfunction + "_results_cifar10d_"+str(SEED)+".txt", "a+") as f:
            f.write("gamma: " + str(gamma) + ", margin: " + str(margin) + ", train_auc: " +
                    str(train_auc) + ", test_auc: " + str(val_auc) + "\n")
        # clear memory
        del model
        del optimizer
        torch.cuda.empty_cache()
