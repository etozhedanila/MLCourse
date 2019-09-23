from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np 
import pandas as pd 

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism','sci.space'])

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(newsgroups.data)
words = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

print('start fit')
gs.fit(x, newsgroups.target)
print('end fit')

c = gs.best_estimator_.C

clf = SVC(C=c, kernel='linear', random_state=241)

fitted_clf = clf.fit(x, newsgroups.target)

weights = []
for elem in fitted_clf.coef_.T:
	weights.append(abs(elem[0]))

d = dict(zip(words, weights))

words=pd.DataFrame(data=d,index=[0]).transpose()
words.columns = ['weights']

words.sort_values(['weights'], ascending=False, inplace=True)

topWords = pd.DataFrame(data=words.head(10).index.values)
topWords.columns = ['topWords']
topWords.sort_values(['topWords'], inplace=True)

print(' '.join(topWords['topWords']))






