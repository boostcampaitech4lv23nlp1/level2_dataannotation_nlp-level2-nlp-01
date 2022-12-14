
import sklearn
import numpy as np
import pickle as pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        '관계없음', 
        '곤충:천적',
        '곤충:공생',
        '곤충:별칭',
        '곤충:특성',
        '곤충:서식지',
        '곤충:발생시기',
        '곤충:먹이',
        '곤충:개체수',
        '곤충:하위분류',
        '생물분류:특징',
        '생물분류:개체수',
        '생물분류:하위분류',
        '생물분류:서식지'
    ]
    no_relation_label_idx = label_list.index("관계없음")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(14)[labels]
    score = np.zeros((14,))
    for c in range(14):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def re_accuracy_score(probs, labels):

    return accuracy_score(probs, labels)