__all__ = ['accuracy','auroc']
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_curve,roc_auc_score,roc_curve
np.random.seed(0)

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
    outputs=softmax(outputs,axis=1) # logits
    preds=np.argmax(outputs,axis=1)
    true=np.equal(targets,preds).astype(int)
    outputs=np.max(outputs,axis=1)
    try:
        roc_auc_value=roc_auc_score(true,outputs)
        np.savetxt("outputs.txt",outputs)
        np.savetxt("targets.txt",targets)
    except:
        roc_auc_value=0.0
    tpr,fpr,_=roc_curve(true,outputs)
    return {'tpr':tpr,'fpr':fpr,'auc':roc_auc_value}
 