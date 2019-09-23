import numpy as np 

#создаем 2 единичные матрицы 3х3
a = np.eye(3)
b = np.eye(3)


print(np.vstack((a,b)))		#объединяем матрицы a и b в одну вертикально