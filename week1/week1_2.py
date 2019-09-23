import numpy as np

Z = np.array([[4, 5, 0], 
             [1, 9, 3],              
             [5, 1, 1],
             [3, 3, 3], 
             [9, 9, 9], 
             [4, 7, 1]])

sumMatrix = np.sum(Z, axis=1)		#получаем матрицу(вектор), состоящую из сумм элементов строк матрицы Z

result = sumMatrix > 10				#если элемент sumMatrix больше 10 - соответствующий элемент result = True, иначе False

print(np.nonzero(result))			#получаем индексы элементов со значением True