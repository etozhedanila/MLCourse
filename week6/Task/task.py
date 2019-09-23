from skimage.io import imread
from skimage import img_as_float
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from math import log10
import numpy as np


def psnr(X_true, X):
	mse = 0
	for i in range(len(X_true)):
		for j in range(len(X_true[0])):
			mse += ((X_true[i][j] - X[i][j]) ** 2)
	mse /= (len(X_true) * len(X_true[0]))
	return 10 * log10(1 / mse)


def create_image_matrix(image):
	image_matrix = []
	for i in range(len(image)):
		for j in range(len(image[i])):
			image_matrix.append(image[i][j])
	return image_matrix


image = imread('parrots.jpg')
image = img_as_float(image)

n = len(image[0])

image_matrix = create_image_matrix(image)

X_train = pd.DataFrame(data=image_matrix, columns=['R', 'G', 'B']) 


for i in range(1, 21):
	X_tmp = X_train.copy()
	kmeans = KMeans(init='k-means++', random_state=241, n_clusters=i).fit(X_train)
	
	X_tmp['cluster']=kmeans.labels_
	X_tmp.set_index('cluster', inplace=True)

	X_1 = X_tmp.copy()
	X_2 = X_tmp.copy()

	for cluster in range(i):
		for column in X_tmp.columns:
			mean = X_1.loc[cluster, column].mean()
			median = X_2.loc[cluster, column].median()

			X_1.loc[cluster, column] = mean
			X_2.loc[cluster, column] = median

	psnr_mean = psnr(np.array(X_tmp), np.array(X_1))
	psnr_median = psnr(np.array(X_tmp), np.array(X_2))

	if psnr_median > 20:
		print(i, "psnr_median: ", psnr_median)
		break
	
	if psnr_mean > 20:
		print(i, "psnr_mean: ", psnr_mean)
		break

