from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as prc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import numpy as np


def score(y_true, y_prob, y_pred):

    select = ~y_true.isna()
    y_prob = y_prob[select]
    y_true = y_true[select]
    y_pred = y_pred[select]  # .map({'NR':0, 'R':1})

    if len(y_true.unique()) == 1:
        roc = np.nan
        prc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
        prc = prc_auc_score(_recall, _precision)

    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)

    return roc, prc, f1, acc


def score2(y_true, y_prob, y_pred):

    select = ~y_true.isna()
    y_prob = y_prob[select]
    y_true = y_true[select]
    y_pred = y_pred[select]  # .map({'NR':0, 'R':1})

    if len(y_true.unique()) == 1:
        roc = np.nan
        prc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
        prc = prc_auc_score(_recall, _precision)

    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return roc, prc, f1, acc, mcc


def score3(y_true, y_prob, y_pred):
    """
    Calculate evaluation metrics including ROC-AUC, PRC-AUC, F1, Accuracy, MCC, FPR, and FNR.

    Parameters:
    y_true (array-like): True binary labels.
    y_prob (array-like): Predicted probabilities for the positive class.
    y_pred (array-like): Predicted binary labels.

    Returns:
    tuple: roc, prc, f1, acc, mcc, fpr, fnr
    """
    # Filter non-NA values
    select = ~y_true.isna()
    y_prob = y_prob[select]
    y_true = y_true[select]
    y_pred = y_pred[select]  # .map({'NR':0, 'R':1})

    # Initialize ROC-AUC and PRC-AUC
    if len(y_true.unique()) == 1:
        roc = np.nan
        prc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
        prc = prc_auc_score(_recall, _precision)

    # Calculate F1, Accuracy, and MCC
    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Calculate Confusion Matrix and derive FPR and FNR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # False Negative Rate

    return roc, prc, f1, acc, mcc, fpr, fnr
