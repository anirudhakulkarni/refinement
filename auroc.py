import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
all_outputs=np.genfromtxt("all_outs.csv", delimiter=",")
all_targets=np.genfromtxt("all_tars.csv", delimiter=",").astype(int)
all_outputs=all_outputs/all_outputs.sum(axis=1).reshape(len(all_outputs),1)
# print(all_outputs)
# print(all_targets)
roc_auc = dict()
roc_auc["ovr"]=roc_auc_score(all_targets,all_outputs,multi_class='ovr')
# print(auroc)
roc_auc["ovo"]=roc_auc_score(all_targets,all_outputs,multi_class='ovo')

fpr = dict()
tpr = dict()
all_targets_one = np.zeros((all_targets.size, all_targets.max()+1))
all_targets_one[np.arange(all_targets.size),all_targets] = 1
# print(all_targets_one)
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(all_targets_one[:,i], all_outputs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_one.ravel(), all_outputs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 10

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



print(roc_auc)