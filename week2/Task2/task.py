import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import sklearn.preprocessing as sp
import sklearn.neighbors as sn
import sklearn.model_selection as sm


boston_data = load_boston()

data = sp.scale(boston_data.data)

kf = sm.KFold(n_splits=5,shuffle=True,random_state=42)

max_mean = 0
max_mean_p = 0
for p in np.linspace(1,10, num=200):
	knr = sn.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p )
	cvs = sm.cross_val_score(knr, data, boston_data.target, cv=kf, scoring='neg_mean_squared_error')
	mean = cvs.mean()
	if mean > max_mean or max_mean_p == 0:
		max_mean = mean
		max_mean_p = p
print(max_mean_p, max_mean)



