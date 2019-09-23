import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

target = data['Rings']

X = data.iloc[:, :8]

cv = KFold(n_splits=5, random_state=1, shuffle=True)
for i in range(1,51):
	clf = RandomForestRegressor(random_state=1, n_estimators=i)
	clf.fit(X, target)
	
	cv_score = cross_val_score(clf, X, target, scoring=make_scorer(r2_score), cv=cv)
	cv_score_mean = cv_score.mean()
	if cv_score_mean > 0.52:
		print(cv_score_mean, i)
		break


