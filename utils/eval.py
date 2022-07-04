__all__ = ['accuracy', 'auroc', 'get_all_metrics']
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score, roc_curve
import sklearn.metrics as metrics

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
    outputs = softmax(outputs, axis=1)  # logits
    preds = np.argmax(outputs, axis=1)
    true = np.equal(targets, preds).astype(int)
    outputs = np.max(outputs, axis=1)
    try:
        roc_auc_value = roc_auc_score(true, outputs)
    except:
        roc_auc_value = 0.0
        np.savetxt("outputs.txt", outputs)
        np.savetxt("targets.txt", targets)
    tpr, fpr, _ = roc_curve(true, outputs)
    return {'tpr': tpr, 'fpr': fpr, 'auc': roc_auc_value}
    # another auc calculation way (classwise individual and then mean) : https://github.com/MayeshMohapatra/ChestXRayClassification/blob/main/chexpert-densenet121.ipynb

def get_all_metrics(targets, outputs):
    # return dictionary with tpr, fpr, auroc, aupr-success, aupr-error, aurc, eaurc, fpr-at-95
    outputs = softmax(outputs, axis=1)  # logits
    preds = np.argmax(outputs, axis=1)
    correct = np.equal(targets, preds).astype(int)
    outputs_max = np.max(outputs, axis=1)
    aurc, eaurc = calc_aurc_eaurc(outputs, correct)
    tpr, fpr, _ = roc_curve(correct, outputs_max)
    aupr_success, aupr_err, fpr_at_95 = calc_fpr_aupr(outputs, correct)
    try:
        roc_auc_value = roc_auc_score(correct, outputs_max)
    except:
        roc_auc_value = 0.0
        np.savetxt("outputs.txt", outputs)
        np.savetxt("targets.txt", targets)
    return {'tpr': tpr, 'fpr': fpr, 'auroc': roc_auc_value, 'aupr-success': aupr_success, 'aupr-error': aupr_err, 'aurc': aurc, 'eaurc': eaurc, 'fpr-at-95': fpr_at_95}


# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(
        zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    # aupr-error
    aupr_err = metrics.average_precision_score(
        -1 * correctness + 1, -1 * softmax_max)
    # aupr-success
    aupr_success = metrics.average_precision_score(correctness, softmax_max)

    return aupr_success, aupr_err, fpr_in_tpr_95


# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
# https://arxiv.org/pdf/1805.08206.pdf
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc
