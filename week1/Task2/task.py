import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).

data = pd.read_csv("titanic.csv", index_col="PassengerId")

data = data[["Survived","Pclass", "Fare", "Age", "Sex"]]

data = data.dropna(axis=0)		#удаляем строки содержащие NaN

y = data["Survived"]

data = data.drop(["Survived"], axis=1)


for index, sex in enumerate(data["Sex"]):
	if sex == "male":
		data.iloc[index,3] = 0
		
	else:
		data.iloc[index,3] = 1

print(data)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, y)

importances = clf.feature_importances_

print(importances)
