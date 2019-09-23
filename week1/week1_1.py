import numpy as np 

mu, sigma = 1, 10

x = np.random.normal(loc=mu,scale=sigma,size=(1000,50))		#матрица  размера 1000 на 50, состоящая из рандомных элементов принадлежащих распределению N(1,100)
 
mid = np.mean(x, axis=0)		#среднее значение по столбцу

std = np.std(x, axis=0)			#стандартное отклонение по столбцу

x = (x-mid/std)

print(x)

