'''
Logistic regression upper bound on performance
'''

import argparse
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn import cross_validation
import bayeslex

parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('--vocab_size',default=50000,type=int)

args = parser.parse_args()
y,x,vocab = bayeslex.loadData(args.prefix,args.vocab_size)

clf = LogisticRegressionCV()
aucs = cross_validation.cross_val_score(clf,x,y>0,cv=5,scoring='roc_auc')
print aucs.mean()
