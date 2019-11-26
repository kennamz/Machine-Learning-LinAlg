import numpy as np
from numpy import linalg as LA
from gradient_descent_step import gradient_descent_step
from fraction_wrong import fraction_wrong
from loss import loss

def gradient_descent(A, b, w, sigma, T):
    for i in range(T):
        w = gradient_descent_step(A, b, w, sigma)
        if i % 5000 == 0:
            print("Fraction wrong:", fraction_wrong(A, b, w))
            print("Loss:", loss(A, b, w))
    return w

