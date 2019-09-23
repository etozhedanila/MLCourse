import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')


full_description = data['FullDescription']
salary_normalized = data['SalaryNormalized']

full_description_test = data_test['FullDescription']

full_description = full_description.replace('[^a-zA-Z0-9]', ' ', regex = True)
full_description_test = full_description_test.replace('[^a-zA-Z0-9]', ' ', regex = True)

for i in range(len(full_description)):
	full_description[i] = full_description[i].lower()

for i in range(len(full_description_test)):
	full_description_test[i] = full_description_test[i].lower()

vectorizer = TfidfVectorizer(min_df=5)

full_description = vectorizer.fit_transform(full_description)
full_description_test = vectorizer.transform(full_description_test)

data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()

X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([full_description, X_train_categ])
X_test = hstack([full_description_test, X_test_categ])

clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, salary_normalized)

predictions = clf.predict(X_test)
for prediction in predictions:
	print(round(prediction, 2))