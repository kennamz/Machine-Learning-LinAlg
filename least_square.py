import numpy as np
#from numpy import linalg as LA
from read_training_data import read_training_data
from gradient_descent import gradient_descent
from fraction_wrong import fraction_wrong
from loss import loss

A, b = read_training_data("train.data")
A = A[...,1:31]
b = b[...,1]
AT = A.transpose()

least_square = np.linalg.inv(AT*A)*AT*b
print(least_square)

print(fraction_wrong(A, b, least_square))
print(loss(A, b, least_square))
