import numpy as np
#from numpy import linalg as LA
from read_training_data import read_training_data
from gradient_descent import gradient_descent
from fraction_wrong import fraction_wrong
from loss import loss

A, b = read_training_data("train.data")
A = A[...,1:31]
b = b[...,1]
w = np.ones((30, 1))/2
least_square = np.matrix([[-8.69922200e-01],
 [ 2.43139450e-02],
 [ 6.26796038e-02],
 [ 3.27462015e-03],
 [ 8.79030463e+00],
 [-1.74714768e+00],
 [ 2.02847615e-01],
 [ 6.50644947e+00],
 [-5.06176015e+00],
 [-4.91675449e+01],
 [ 9.56592244e-01],
 [ 8.20527005e-02],
 [ 7.94307879e-03],
 [-4.97690954e-03],
 [ 2.78419378e+01],
 [-3.30151884e+00],
 [-4.98595672e+00],
 [ 1.63188873e+01],
 [-1.03162901e+01],
 [ 2.13321684e+01],
 [ 4.08605799e-01],
 [ 3.34571693e-03],
 [ 6.77876803e-04],
 [-2.51073512e-03],
 [-4.53136916e+00],
 [-5.90110921e-01],
 [ 7.19368538e-01],
 [ 2.15896728e+00],
 [ 3.80346694e+00],
 [ 1.22984246e+01]])
hyp = np.matrix([[ 0.26420674],
 [-0.00921931],
 [-0.20270327],
 [ 0.00967819],
 [ 0.4966312 ],
 [ 0.5123027 ],
 [ 0.5131531 ],
 [ 0.51381565],
 [ 0.48640186],
 [ 0.49279019],
 [ 0.46213978],
 [ 0.10319931],
 [ 0.15409686],
 [-0.01378782],
 [ 0.49895125],
 [ 0.49500919],
 [ 0.48467361],
 [ 0.49873741],
 [ 0.49452338],
 [ 0.49864858],
 [ 0.38480811],
 [ 0.01118984],
 [ 0.03657246],
 [-0.00337195],
 [ 0.49874736],
 [ 0.54998982],
 [ 0.5394955 ],
 [ 0.52692426],
 [ 0.4947247 ],
 [ 0.49704665]])
sigma = .001
T = 500000
final = gradient_descent(A, b, w, sigma, T)
#print(final)


validate_A, validate_b = read_training_data("validate.data")
validate_A = validate_A[...,1:31]
validate_b = validate_b[...,1]
print(fraction_wrong(validate_A, validate_b, final))
#print(fraction_wrong(validate_A, validate_b, least_square))
#print(loss(A, b, least_square))
#print(loss(validate_A, validate_b, least_square))


