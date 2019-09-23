import sklearn.model_selection as sm
import sklearn.neighbors as sn
import pandas as pd 
import numpy as np
import sklearn.preprocessing as sp

data = pd.read_csv("wine.data", sep=",", header=None)

classes = data[0]
#classes = np.array(classes)


attributes = data.iloc[:,1:]
attributes = sp.scale(attributes)
#attributes = np.array(attributes)


kf = sm.KFold(n_splits=5, shuffle=True, random_state=42)

cvs = sm.cross_val_score(sn.KNeighborsClassifier(), attributes, classes, cv=kf, scoring='accuracy')
print(cvs)
mean = cvs.mean()
print(mean)

#1 2
# max = mean
# k_max = 5
# for k in range(1,51):
# 	tmp = sm.cross_val_score(sn.KNeighborsClassifier(n_neighbors=k), attributes, classes, cv=kf)
# 	tmp_mean = tmp.mean()
# 	if tmp_mean > max:
# 		max = tmp_mean
# 		k_max = k

# print(k_max, max)

#3 4
max = mean
k_max = 5
for k in range(1,51):
	tmp = sm.cross_val_score(sn.KNeighborsClassifier(n_neighbors=k), attributes, classes, cv=kf)
	tmp_mean = tmp.mean()
	if tmp_mean > max:
		max = tmp_mean
		k_max = k

print(k_max, max)




