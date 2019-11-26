import numpy as np
from numpy import linalg as LA
from find_grad import find_grad

def gradient_descent_step(A, b, w, sigma):
    grad = find_grad(A, b, w)
    grad_with_step = sigma*grad
    #print(w)
    #print(grad_with_step)
    new_hypothesis = w - grad_with_step
    return new_hypothesis


