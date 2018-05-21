#MPCA Merlin

import numpy as np
from matplotlib.mlab import PCA


"""
file = open("C:\Users\\Tiphaine Casy\\Documents\\GitHub\\Merlin\\data.csv", "r")
datas = file.readlines()

print datas
"""


# Principal Component Analysis
from numpy import array
from sklearn.decomposition import PCA
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)