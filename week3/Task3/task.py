import pandas as pd 
from sklearn.metrics import roc_auc_score
import numpy as np
from math import sqrt
from math import exp

def distance(last_w, curr_w):
	return sqrt((last_w[0]-curr_w[0]) ** 2 + (last_w[1]-curr_w[1]) ** 2)
	
def gradient_descent(x, y, C=0):
	last_w = [0, 0]
	curr_w = [0, 0]
	k = 0.1
	eps = 0.00001
	max_iteration_number = 10000
	l = y.count()
	i = 0
	while i < max_iteration_number:
		i += 1
		sum1 = 0
		sum2 = 0
		for i in range(len(y)):
			coef = 1 - 1/( 1 + exp(-y[i]* (last_w[0]*x[i][0] + last_w[1]*x[i][1]) ) )
			sum1 += coef * y[i] * x[i][0]
			sum2 += coef * y[i] * x[i][1]
		curr_w[0] += k * sum1 / l - k * C * last_w[0] 
		curr_w[1] += k * sum2 / l - k * C * last_w[1] 
		
		if distance(last_w, curr_w) < eps: 
			print(i)
			return curr_w
		last_w = curr_w.copy()
	print("max_iteration_number reached")
	return curr_w

def a(x, w):
	return 1 / (1 + exp(-w[0]*x[0] - w[1]*x[1]))


data = pd.read_csv("data-logistic.csv", names=['target', 'attribute1','attribute2'])
y = data['target']
x = data[['attribute1','attribute2']]

x = np.array(x)

weights = [gradient_descent(x, y), gradient_descent(x, y, 10)]

y_test1 = []
y_test2 = []
for elem in x:
	y_test1.append(a(elem, weights[0]))
	y_test2.append(a(elem, weights[1]))

auc1 = roc_auc_score(y, y_test1)
auc2 = roc_auc_score(y, y_test2)
print(round(auc1, 3), round(auc2, 3))