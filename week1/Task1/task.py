import pandas

# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), 
# округлив до двух знаков.
# Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
#  округлив до двух знаков.
# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.
# Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
# Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name). Это задание — типичный пример того, 
# с чем сталкивается специалист по анализу данных. Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. 
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.

data = pandas.read_csv("titanic.csv", index_col="PassengerId")


#1

male, female = data['Sex'].value_counts()

print("male, female = ", male, female)

#2

notSurvived, survived = data["Survived"].value_counts()

print("notSurvived, survived ", notSurvived, survived)

print(round(100 * survived / (notSurvived + survived), 2) , " %" )

#3

all_class = data['Pclass'].value_counts()
all_passangers = all_class.sum()
print("first_class passengers, all passengers = ", all_class[1],all_passangers)
print(round(100 * all_class[1] / all_passangers, 2), " %")

#4

print("avg age", data['Age'].mean())
print("median age", data['Age'].median())

#5

print("pearson corr", data[['SibSp','Parch']].corr())

#6

names = data['Name']

first_female_names = []
for name in names:
	if "Miss" in name:
		
		first_female_names.append(name.split('Miss.')[1].split()[0])
		continue
	if "Mlle" in name:
		
		first_female_names.append(name.split('Mlle.')[1].split()[0])
		continue
	if "Mrs" in name:
		
		index = name.find("(")
		if index != -1:
			first_name = name[index+1:-1].split()[0]
		else:
			first_name = name[name.find("Mrs."):].split()[1]
		first_female_names.append(first_name)
		

first_female_names.sort()
count = 1
max_count = 1
max_popular_name = first_female_names[0]
for i in range(1,len(first_female_names)):
	if first_female_names[i-1] == first_female_names[i]:
		count += 1
	else:
		if count > max_count:
			max_count = count
			max_popular_name = first_female_names[i-1]
		count = 1
print(max_count, max_popular_name)
print(first_female_names)
	
