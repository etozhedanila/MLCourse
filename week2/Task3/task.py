import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

train = pd.read_csv("perceptron-train.csv", names=['target','attribute1','attribute2'])
test = pd.read_csv("perceptron-test.csv", names=['target','attribute1','attribute2'])

target_variable = train['target']
attributes = train[['attribute1', 'attribute2']]

p = Perceptron(random_state=241, max_iter=5, tol=None)

p.fit(attributes, target_variable)
prediction = p.predict(np.array(test[['attribute1', 'attribute2']]))


accuracy_before = sklearn.metrics.accuracy_score(test['target'], prediction)

scaler = StandardScaler()

train_scaled = pd.DataFrame(scaler.fit_transform(attributes), columns=['attribute1','attribute2'])
test_scaled = pd.DataFrame(scaler.transform(test[['attribute1', 'attribute2']]), columns=['attribute1','attribute2'])


p.fit(train_scaled, target_variable)
prediction = p.predict(test_scaled)


accuracy_after = sklearn.metrics.accuracy_score(test['target'], prediction)

difference = round(accuracy_after - accuracy_before, 3)
print(accuracy_after, accuracy_before, difference)