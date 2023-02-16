import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import torch

def auc_m(y_true, y_pred, label1=None, label2=None, freq = False):
    return roc_auc_score(y_true, y_pred,multi_class='ovo')
    print(y_true)
    print(y_pred)
    print(len(y_true))
    print(len(y_pred))
    print(y_true.shape)
    print(y_pred.shape)
    # If y_pred is of format (n_samples,) then we need to reshape it to (n_samples, 1)
    if len(y_true.shape) == 1:
        y_true = np.reshape(y_true, [-1, 1])
    y_true=np.reshape(y_true,[-1,1])
    enc1 = OneHotEncoder()
    enc1.fit(y_true)
    y_true = enc1.transform(y_true).toarray()
    y_pred_shape=np.shape(y_pred)
    if len(y_pred_shape)==1 or y_pred_shape[1]==1:
        y_pred = np.reshape(y_pred, [-1, 1])
        y_pred = enc1.transform(y_pred).toarray()
    def auc_binary(i, j):
        msk = np.logical_or(y_true.argmax(axis=1) == i, y_true.argmax(axis=1) == j)
        return roc_auc_score(y_true[:, i][msk], y_pred[:, i][msk])
    n = y_true.shape[1]
    
    if not freq:
        return np.mean([auc_binary(i, j) for i in range(n) for j in range(n) if i != j])
    else:
        return auc_binary(label1, label2)
        


# taken from https://github.com/Jonathan-Pearce/calibration_library/blob/master/metrics.py

from scipy.special import softmax

class BrierScore():
    def __init__(self) -> None:
        pass

    def loss(self, outputs, targets):
        K = outputs.shape[1]
        one_hot = np.eye(K)[targets]
        probs = softmax(outputs, axis=1)
        return np.mean( np.sum( (probs - one_hot)**2 , axis=1) )


class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        if self.predictions.shape!=labels.shape:
            # change from one-hot to class index
            self.labels = np.argmax(labels, axis=1)
        # print(self.predictions)
        # print(self.labels)
        self.accuracies = np.equal(self.predictions,self.labels)
        # print(self.accuracies)
        # print("Accuraciesu: ", np.count_nonzero(self.accuracies,axis=0))

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop, self.bin_score)
        return sce/self.n_class

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)
    
    
    
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct=correct.contiguous()
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def get_all_metrics(name, output, target, n_bins = 15, logits = False):
    metrics = {}
    metrics[name+'_top1'], metrics[name+'_top5'] = accuracy(output, target, topk=(1,5))
    # convert probability of class 1 to 2d vector with probability of class 0 and 1
    output = np.stack([1 - output, output], axis=1)
    assert output.shape[1] == 2
    metrics[name+'_ece'] = ECELoss().loss(output, target, n_bins, logits)
    metrics[name+'_mce'] = MCELoss().loss(output, target, n_bins, logits)
    metrics[name+'_oel'] = OELoss().loss(output, target, n_bins, logits)
    metrics[name+'_sce'] = SCELoss().loss(output, target, n_bins, logits)
    metrics[name+'_ace'] = ACELoss().loss(output, target, n_bins, logits)
    metrics[name+'_tace'] = TACELoss().loss(output, target, n_bins, logits)
    
    metrics[name+'_auc'] = roc_auc_score(target, output[:,1])
    return metrics
    
