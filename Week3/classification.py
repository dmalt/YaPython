# IPython log file

import numpy as np
import pandas as pd
data = pd.read_csv('classification.csv')
is_tp = np.array(
    [data['true'][i] and data['pred'][i] for i in range(len(data['true']))])

is_tn = np.array(
    [not data['true'][i] and not data['pred'][i] for i in range(len(data['true']))])

is_fp = np.array(
    [not data['true'][i] and data['pred'][i] for i in range(len(data['true']))])

is_fn = np.array(
    [data['true'][i] and not data['pred'][i] for i in range(len(data['true']))])

tp = is_tp.sum()
tn = is_tn.sum()
fp = is_fp.sum()
fn = is_fn.sum()

answer1_str = str(tp) + ' ' + str(fp) + ' ' + str(fn) + ' ' + str(tn)
fname1 = 'class1.txt'

f1 = open(fname1, 'w')
f1.write(answer1_str)


from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score



_accuracy = float(tp + tn) / (tp + tn + fp + fn)
_precision = tp / float(tp + fp)
_recall = tp / float(tp + fn)
_F = 2 * float(_precision * _recall) / (_precision + _recall)

accuracy  = round(accuracy_score(data['true'], data['pred']), 2)
precision = round(precision_score(data['true'], data['pred']), 2)
recall    = round(recall_score(data['true'], data['pred']), 2)
f1        = round(f1_score(data['true'], data['pred']), 2)

answer2_str = str(accuracy) + ' ' + str(precision) + ' '\
            + str(recall) + ' ' + str(f1)
# print accuracy == _accuracy
# print precision == _precision
# print recall == _recall
# print f1 == _F

fname2 = 'class2.txt'
f2 = open(fname2, 'w')
f2.write(answer2_str)


scores = pd.read_csv('scores.csv')
from sklearn.metrics import roc_auc_score

y_true = scores['true']
roc_logreg = round(roc_auc_score(y_true, scores['score_logreg']), 2)
roc_svm    = round(roc_auc_score(y_true, scores['score_svm']), 2)
roc_knn    = round(roc_auc_score(y_true, scores['score_knn']), 2)
roc_tree   = round(roc_auc_score(y_true, scores['score_tree']), 2)


answer3_str = str(roc_logreg) + ' ' + str(roc_svm) + ' '\
            + str(roc_knn) + ' ' + str(roc_tree)
print answer3_str

roc_list = [roc_logreg, roc_svm, roc_knn, roc_tree]
ind = roc_list.index(max(roc_list)) + 1

fname3 = 'class3.txt'
f3 = open(fname3, 'w')
f3.write(scores.columns[ind])

# --------------------------------------------- #

from sklearn.metrics import precision_recall_curve
max_precisions = {}
for classifier in scores.columns[1:]:
    this_scores = scores[classifier]
    precision, recall, thresholds = precision_recall_curve(y_true, this_scores)
    max_precisions[classifier] = precision[recall > 0.7].max()
fname4 = 'class4.txt'
f4 = open(fname4, 'w')
f4.write(max(max_precisions))
