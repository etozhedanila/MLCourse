import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

prices = pd.read_csv('close_prices.csv')
djia_index = pd.read_csv('djia_index.csv')

djia = djia_index['^DJI']

X = prices.iloc[:, 1:]

pca = PCA(n_components=10)
pca.fit(X)

#1
sum = 0
for (index, elem) in enumerate(pca.explained_variance_ratio_):
	sum += elem
	if sum >= 0.9:
		print(index + 1)
		break

#2
transformed_X = pca.transform(X)[:,0]
print(np.corrcoef(transformed_X, djia))


#3
component = pca.components_
max_weight = max(component[0])
index = np.where(component == max_weight)[1][0]
print(prices.columns[index+1])




