import pandas as pd 
from sklearn.svm import SVC

data = pd.read_csv("svm-data.csv", names=['target', 'attribute1', 'attribute2'])

target_var = data['target']

attributes = data[['attribute1', 'attribute2']]

clf = SVC(kernel='linear', C=100000, random_state=241)

clf.fit(attributes, target_var)

print(clf.support_)