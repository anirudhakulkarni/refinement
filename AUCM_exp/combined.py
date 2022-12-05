from libauc.losses import AUCMLoss, CrossEntropyLoss, SupConLoss
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
lossfunction = sys.argv[1]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class myBatchNorm2d(torch.nn.Module):
    def __init__(self, input_size = None , epsilon = 1e-3, momentum = 0.99):
        super(myBatchNorm2d, self).__init__()
        # assert input_size, print('Missing input_size parameter.')
        
        # Batch mean & var must be defined during training
        self.mu = torch.zeros(1, input_size)
        self.var = torch.ones(1, input_size)
        
        # For numerical stability
        self.epsilon = epsilon
        
        # Exponential moving average for mu & var update 
        self.it_call = 0  # training iterations
        self.momentum = momentum # EMA smoothing
        
        # Trainable parameters
        self.beta = torch.nn.Parameter(torch.zeros(1, input_size))
        self.gamma = torch.nn.Parameter(torch.ones(1, input_size))
        
        # Batch size on which the normalization is computed
        self.batch_size = 0

        
    def forward(self, x):
        # [batch_size, input_size]
        
        self.it_call += 1
        
        if self.training :
            
            if( self.batch_size == 0 ):
                # First iteration : save batch_size
                self.batch_size = x.shape[0]
            
            # Training : compute BN pass
            batch_mu = (x.sum(dim=0)/x.shape[0]).unsqueeze(0) # [1, input_size]
            batch_var = (x.var(dim=0)/x.shape[0]).unsqueeze(0) # [1, input_size]
            
            x_normalized = (x-batch_mu)/torch.sqrt(batch_var + self.epsilon) # [batch_size, input_size]
            x_bn = self.gamma * x_normalized + self.beta # [batch_size, input_size]
            
            
            # Update mu & std 
            if(x.shape[0] == self.batch_size):
                running_mu = batch_mu
                running_var = batch_var
            else:
                running_mu = batch_mu*self.batch_size/x.shape[0]
                running_var = batch_var*self.batch_size/x.shape[0]
 
            self.mu = running_mu * (self.momentum/self.it_call) + \
                            self.mu * (1 - (self.momentum/self.it_call))
            self.var = running_var * (self.momentum/self.it_call) + \
                        self.var * (1 - (self.momentum/self.it_call))
            
        else:
            # Inference : compute BN pass using estimated mu & var
            if (x.shape[0] == self.batch_size):
                estimated_mu = self.mu
                estimated_var = self.var
            else :
                estimated_mu = self.mu*x.shape[0]/self.batch_size
                estimated_var = self.var*x.shape[0]/self.batch_size
                
            x_normalized = (x-estimated_mu)/torch.sqrt(estimated_var + self.epsilon) # [batch_size, input_size]
            x_bn = self.gamma * x_normalized + self.beta # [batch_size, input_size]
    
        return x_bn # [batch_size, output_size=input_size]
class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        if lossfunction == "supcontrast":
            self.transform_train = TwoCropTransform(transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((crop_size, crop_size), padding=None),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)),
            ]))
            self.transform_test = TwoCropTransform(transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
            ]))
        else:
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
gammas = [ 300]
margins = [ 1.0]

lossfunction = sys.argv[1]
datast=sys.argv[2]
modelname=sys.argv[3]
set_all_seeds(SEED)
if lossfunction=='ce':
    gammas = [-1]
    margins = [ -1]
best=[0,0,0,0]
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
        elif datast=='c2':
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

        # You need to include sigmoid activation in the last layer for any customized models!
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
        elif lossfunction == 'supcontrast':
            Loss = SupConLoss()
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

            if lossfunction=='aucm' and (epoch == 50 or epoch == 75):
                # decrease learning rate by 10x & update regularizer
                optimizer.update_regularizer(decay_factor=10)

            train_pred = []
            train_true = []
            model.train()
            for data, targets in trainloader:
                loss=0
                if lossfunction == 'supcontrast':
                    data = torch.cat([data[0], data[1]], dim=0)
                    data = data.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    # print("supcontrast")
                    bsz = targets.shape[0]
                    y_pred=model(data)
                    f1, f2 = torch.split(y_pred, [bsz, bsz], dim=0)
                    y_pred = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss = Loss(y_pred, targets)
                    y_pred = y_pred.contiguous().view(-1, 1)
                    # take only first half of the y_pred
                    # print(y_pred)
                    # y_pred=y_pred[:y_pred.shape[0]//2]
                    # print(y_pred)
                else:
                    data, targets = data.cuda(), targets.cuda()
                    y_pred = model(data)
                    if lossfunction=='aucm':
                        y_pred = torch.sigmoid(y_pred)
                    loss = Loss(y_pred, targets)
                    # l2 normalization
                    # y_pred = torch.nn.modules.normalize(y_pred, p=2, dim=1)
                    # y_pred = torch.nn.BatchNorm1d(1).cuda()(y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_pred.append(y_pred.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            # print(train_true.shape)
            # print(train_pred.shape)
            # print(train_pred)            
            train_auc = roc_auc_score(train_true, train_pred)

            model.eval()
            test_pred = []
            test_true = []
            for j, data in enumerate(testloader):
                test_data, test_targets = data
                test_data = test_data.cuda()
                y_pred = model(test_data)
                if lossfunction=='aucm':
                    y_pred = torch.sigmoid(y_pred)
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(test_targets.numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc = roc_auc_score(test_true, test_pred)
            model.train()
            if lossfunction=='ce':
                scheduler.step()
            # print learning rate
            # for param_group in optimizer.param_groups:
            #     print("learning rate: ", param_group['lr'])
            # eces = ECELoss().loss(test_pred, test_true, n_bins=15)
            # cces = SCELoss().loss(test_pred, test_true, n_bins=15)
            # print('Epoch: {} | Train_loss: {:4f} | Train AUC: {:.4f} | Val AUC: {:.4f} | ECE: {:.4f} | SCE: {:.4f}'.format(
            #     epoch, loss.item(), train_auc, val_auc, eces, cces))
            # print results
            if best==[0,0,0,0] or val_auc>best[3]:
                best = [gamma, margin, train_auc, val_auc]
                torch.save(model.state_dict(), lossfunction + "_model_"+datast+modelname+str(SEED)+".pth")

            print("epoch: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}".format(
                epoch, loss.item(), train_auc, val_auc))

        # save results
        with open(lossfunction + "_results_"+datast+modelname+str(SEED)+".txt", "a+") as f:
            # f.write("gamma: " + str(gamma) + ", margin: " + str(margin) + ", train_auc: " +
            #         str(train_auc) + ", test_auc: " + str(val_auc) + "\n")
            f.write("gamma: " + str(best[0]) + ", margin: " + str(best[1]) + ", train_auc: " +
                    str(best[2]) + ", test_auc: " + str(best[3]) + "\n")
        # clear memory
        del model
        del optimizer

        torch.cuda.empty_cache()