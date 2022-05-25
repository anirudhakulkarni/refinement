__all__ = ['accuracy','auroc']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    maxk = min(output.shape[-1], maxk)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def auroc(targets, outputs):
    import numpy as np
    import matplotlib.pyplot as plt
    # from sklearn.preprocessing import normalize
    from scipy.special import softmax
    from sklearn.metrics import roc_curve, auc,roc_auc_score,roc_curve
    outputs=softmax(outputs,axis=1) # logits
    preds=np.argmax(outputs,axis=1)
    true=np.equal(targets,preds).astype(int)
    print(true)
    targets=true
    outputs=np.max(outputs,axis=1)
    np.savetxt("outputs.csv",outputs,delimiter=",")
    np.savetxt("targets.csv",targets,delimiter=",")
    # from sklearn.metrics import roc_auc_score, roc_curve, auc
    # outputs-=outputs.min()
    # outputs=normalize(outputs)
    # print(outputs.sum(axis=1))
    # outputs/=outputs.astype(float).max()
    # outputs=outputs.astype(float)/outputs.astype(float).sum(axis=1).reshape(len(outputs),1)
    np.savetxt("outputs_one.csv",outputs,delimiter=",")
    roc_auc_dic = dict()
    # roc_auc["ovr"]=roc_auc_score(targets,outputs,multi_class='ovr')
    # roc_auc["ovo"]=roc_auc_score(targets,outputs,multi_class='ovo')
    roc_auc_dic["single"]=roc_auc_score(targets,outputs)
    tpr,fpr,_=roc_curve(targets,outputs)
    plt.figure()
    plt.plot(tpr,fpr)
    # plt.show()
    plt.savefig("/DATA/scratch/anirudha/MDCA-Calibration-main/auroc-4.png")
    # fpr = dict()
    # tpr = dict()
    # targets_one = np.zeros((targets.size, targets.max()+1))
    # targets_one[np.arange(targets.size),targets] = 1
    # print(targets_one)
    # for i in range(10):
    #     fpr[i], tpr[i], _ = roc_curve(targets_one[:,i], outputs[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(targets_one.ravel(), outputs.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(10):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # # Finally average it and compute AUC
    # mean_tpr /= 10

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc_dic
 