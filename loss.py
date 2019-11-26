import numpy as np
from numpy import linalg as LA

def loss(A, b, w):
    # ||(A dot w - b)||^2
    product = A.dot(w)
    result = product - b
    #print(result)
    magnitude = LA.norm(result)
    #print(magnitude)
    return magnitude**2
    







#A = np.matrix([[1,-5],[-6,4],[3,2]])
#b = np.matrix([[1], [-1], [1]])
#w = np.matrix([[2], [-5]])
#print(loss(A, b, w))
