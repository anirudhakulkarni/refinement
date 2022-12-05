from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG
from libauc.models import resnet20 as ResNet20, densenet121
from libauc.datasets import CAT_VS_DOG, CIFAR10, CIFAR100
from libauc.utils import ImbalancedDataGenerator
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import sys

def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            y_pred: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements agaionst (subsampled) negative elements
        if ln_pos>0 :
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + gamma
            l2 = diff2[diff2>0]
            m2 = l2 * l2
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()

        if (torch.sum(m2)+torch.sum(m3))!=0 :
            res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
        else:
            res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2


def epoch_update_gamma(y_true,y_pred, epoch=-1,delta=2):
    """
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    DELTA = delta
    SUB_SAMPLE_SIZE = 2000.0
    pos = y_pred[y_true==1]
    neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]
    pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE/cap_pos]
    neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE/cap_neg]
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    diff = neg_expand - pos_expand
    ln_All = diff.shape[0]
    Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0]-1
    diff_neg = -1.0 * diff[diff<0]
    diff_neg = diff_neg.sort()[0]
    ln_neg = diff_neg.shape[0]-1
    ln_neg = max([ln_neg, 0])
    left_wing = int(ln_Lp*DELTA)
    left_wing = max([0,left_wing])
    left_wing = min([ln_neg,left_wing])
    default_gamma=torch.tensor(0.2, dtype=torch.float).cuda()
    if diff_neg.shape[0] > 0 :
       gamma = diff_neg[left_wing]
    else:
       gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
    L1 = diff[diff>-1.0*gamma]
    ln_L1 = L1.shape[0]
    if epoch > -1 :
        return gamma
    else :
        return default_gamma



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
imratio = 0.01  
lr = 0.1
weight_decay = 1e-4

gammas = [100, 300, 500, 700, 1000]
margins = [0.1, 0.3, 0.5, 0.7, 1.0]
gammas=[300]
margins=[1]
lossfunction = sys.argv[1]
datast=sys.argv[2]
modelname=sys.argv[3]
set_all_seeds(SEED)
if lossfunction=='ce':
    gammas = [-1]
    margins = [ -1]

# Tune hyperparameters, e.g., gammas and margins
for gamma in gammas:
    for margin in margins:
        # dataloader
        if datast=='cifar10':
            train_data, train_targets = CIFAR10(root='../../MDCA-Calibration-main/data')
            test_data, test_targets = CIFAR10(root='../../MDCA-Calibration-main/data',train=False)
        elif datast=='cifar100':
            train_data, train_targets = CIFAR100(root='../../MDCA-Calibration-main/data')
            test_data, test_targets = CIFAR100(root='../../MDCA-Calibration-main/data',train=False)
        elif datast=='catvsdog':
            train_data, train_targets = CAT_VS_DOG(root='../../MDCA-Calibration-main/data')
            test_data, test_targets = CAT_VS_DOG(root='../../MDCA-Calibration-main/data',train=False)
        else:
            print('wrong dataset')
            exit(0)
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

        if modelname=='resnet20':
            model = ResNet20(pretrained=False, last_activation=None, num_classes=1)
        elif modelname=='densenet121':
            model = densenet121(pretrained=False, last_activation=None, num_classes=1)
        else:
            print('wrong model')
            exit(0)
        model = model.cuda()
        if lossfunction == 'aucm':
            Loss = AUCMLoss()
        elif lossfunction == 'ce':
            Loss = CrossEntropyLoss()
        elif lossfunction == 'roc':
            Loss = roc_star_loss
        else:
            print("loss function not supported")
            exit()
        print("gamma: ", gamma)
        print("margin: ", margin)
        if lossfunction == 'aucm' :
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
                if epoch>0:
                    if lossfunction == 'aucm':
                        y_pred = torch.sigmoid(y_pred)
                    if lossfunction=='roc':
                        loss = Loss(targets,y_pred,epoch_gamma,last_whole_y_t,last_whole_y_pred)
                    else:
                        loss = Loss(y_pred, targets)
                else:
                    loss = CrossEntropyLoss()(y_pred, 1.0*targets)
                if epoch>50:
                    Loss=roc_star_loss
                    lossfunction='roc'
                # loss = Loss(y_pred, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                train_pred.append(y_pred.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_auc = roc_auc_score(train_true, train_pred)
            last_whole_y_t = torch.tensor(train_true).cuda()
            last_whole_y_pred = torch.tensor(train_pred).cuda()
            # epoch_gamma = epoch_update_gamma(last_whole_y_t, last_whole_y_pred, epoch,2)
            # print(epoch_gamma)
            epoch_gamma=0.3
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
            print("epoch: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}".format(
                epoch, loss.item(), train_auc, val_auc))

        # save results
        with open(lossfunction + "_results_"+datast+modelname+str(SEED)+".txt", "a+") as f:
            f.write("gamma: " + str(gamma) + ", margin: " + str(margin) + ", train_auc: " +
                    str(train_auc) + ", test_auc: " + str(val_auc) + "\n")
        # clear memory
        del model
        del optimizer

        torch.cuda.empty_cache()