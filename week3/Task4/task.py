import pandas as pd 
import numpy as np
import sklearn.metrics as sm


data = pd.read_csv("classification.csv")

true = data['true']
pred = data['pred']

#1

tp, tn = 0, 0
fp, fn = 0, 0 
for i in range(len(true)):
	if true[i] == pred[i]:
		if true[i] == 1:
			tp += 1
		if true[i] == 0:
			tn += 1
	if true[i] != pred[i]:
		if true[i] == 1:
			fn += 1
		if true[i] == 0:
			fp += 1
print(tp, fp, fn, tn) 

#2

accuracy = sm.accuracy_score(true, pred)
print(round(accuracy,2))

precision = sm.precision_score(true, pred)
print(round(precision,2))

recall = sm.recall_score(true, pred)
print(round(recall,2))

f_measure = sm.f1_score(true,pred)
print(round(f_measure,2))

#3

scores = pd.read_csv("scores.csv")

true = scores['true']
scores_arr = [scores['score_logreg'], scores['score_svm'], scores['score_knn'], scores['score_tree']]

max_area = 0
max_area_column = ''
for score in scores_arr:
	area = sm.roc_auc_score(true, score)
	if area > max_area:
		max_area = area
		max_area_column = score.name

print(max_area, max_area_column)

#4

max_precision = 0
max_precision_column = ''
for score in scores_arr:
	precision, recall, thresholds = sm.precision_recall_curve(true, score)
	for i in range(len(precision)):
		if recall[i] >= 0.7:
			if precision[i] > max_precision:
				max_precision = precision[i]
				max_precision_column = score.name
		else:
			break
print(max_precision ,max_precision_column)