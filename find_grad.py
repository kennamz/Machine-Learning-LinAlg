import numpy as np
from numpy import linalg as LA

def find_grad(A, b, w):
    '''takes as input the training data A, b and a
    hypothesis vector w and returns the value of
    the gradient of L at the point w'''
    product = (A.dot(w) - b)
    result = np.transpose(A)*2*product
    unit = LA.norm(result)
    unit_result = result/unit
    return unit_result
