import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, jaccard_score)

def metrics(true, pred) :
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    true = np.asarray(true.flatten(), dtype=np.int64)
    pred = np.asarray(pred.flatten(), dtype=np.int64)

    acc = accuracy_score(true, pred)
    pre = precision_score(true, pred, average='macro')
    rec = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')

    return acc, f1, pre, rec, iou

def MultiClassSegmentationMetrics(true, pred, num_classes):
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for class_idx in range(num_classes):
        TP_list.append(np.sum((true == class_idx) & (pred == class_idx)))
        FP_list.append(np.sum((true != class_idx) & (pred == class_idx)))
        TN_list.append(np.sum((true != class_idx) & (pred != class_idx)))
        FN_list.append(np.sum((true == class_idx) & (pred != class_idx)))

    return TP_list, FP_list, TN_list, FN_list

def Accuracy_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    accuracy_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]
        TN = cfm_per_class['TN'][class_idx]
        FP = cfm_per_class['FP'][class_idx]

        accuracy_per_class.append((TP + TN + eps) / (TN + FP + TP + FN + eps))

    return accuracy_per_class

def Precision_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    precision_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FP = cfm_per_class['FP'][class_idx]

        precision_per_class.append((TP + eps) / (TP + FP + eps))

    return precision_per_class

def Recall_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    recall_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]

        recall_per_class.append((TP + eps) / (TP + FN + eps))

    return recall_per_class

def F1Score_Calculator(precision_per_class, recall_per_class, num_classes=6, eps=0.0001):
    f1_score_per_class = []

    for class_idx in range(num_classes):
        f1_score_per_class.append(2*(precision_per_class[class_idx] * recall_per_class[class_idx] + eps) / (precision_per_class[class_idx] + recall_per_class[class_idx] + eps))

    return f1_score_per_class


def IoU_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    iou_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FP = cfm_per_class['FP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]

        iou_per_class.append((TP + eps) / (TP + FP + FN + eps))

    return iou_per_class