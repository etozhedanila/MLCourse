import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from math import exp


def sigm(y_pred):
	result_y = []
	for elem in y_pred:
		result_y.append(1 / (1 + exp(-elem)))
	return result_y


data = pd.read_csv('gbm-data.csv')


y = data['Activity']
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

iter_number = 0
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
for rate in learning_rates:
	clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=rate)
	clf.fit(X_train, y_train)

	sdf_train = clf.staged_decision_function(X_train)
	sdf_test = clf.staged_decision_function(X_test)

	score_train = []
	for y_pred in sdf_train:
		score_train.append(log_loss(y_train, sigm(y_pred)))


	score_test = []
	min_loss = 1
	for i, y_pred in enumerate(sdf_test):
		loss = log_loss(y_test, sigm(y_pred))
		score_test.append(loss)
		if rate == 0.2 and loss < min_loss:
			min_loss = loss
			iter_number = i



	if rate == 0.2:
		print(min_loss, iter_number)

	
	plt.figure()
	plt.plot(score_train, 'r', linewidth=3)
	plt.plot(score_test, 'g', linewidth=2)
	plt.legend(['train', 'test'])

plt.show()

clf = GradientBoostingClassifier(n_estimators=iter_number ,random_state=241)
clf.fit(X_train, y_train)
prediction = clf.predict_proba(X_test)
print(round(log_loss(y_test, prediction), 2))

