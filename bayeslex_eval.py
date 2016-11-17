from sklearn.metrics import roc_auc_score
import numpy as np

def threeClassAcc(labels,preds):
    '''
    This is for cases where labels are in {-1,0,1}, and preds are true and false.
    It's something like recall.
    '''
    tp = ((labels > 0) & preds).sum()
    tn = ((labels < 0) & ~preds).sum()
    acc = (tp + tn) / np.abs(labels).sum()
    return acc

def resultString(predictions, y, sysname):
    '''
    predictions should be a number, positive = positive label
    '''
    acc = threeClassAcc(y,predictions>0)
    auc = roc_auc_score(y>0,predictions)
    return "OOO %s:\t%.3f\t%.3f"%(sysname,acc,auc)
