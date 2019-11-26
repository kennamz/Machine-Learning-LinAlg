import numpy as np
from numpy import linalg as LA

def read_training_data(fname):
    """Given a file in appropriate format, and given a set D of features,
    returns the pair (A, b) consisting of
    a P-by-D matrix A and a P-vector b,
    where P is a set of patient identification integers (IDs).

    For each patient ID p,
      - row p of A is the D-vector describing patient p's tissue sample,
      - entry p of b is +1 if patient p's tissue is malignant, and -1 if it is benign.

    The set D of features must be a subset of the features in the data (see text).
    """
    file = open(fname)
    file = file.readlines()
    num_lines = sum(1 for line in file)
    matA = np.zeros((num_lines, 31))
    matb = np.empty((num_lines, 2))
    currentLineNum = 0
    for line in file[0:]:
        line.rstrip()
        row = line.split(",")
        firstEl = [0]
        firstEl[0] = int(row[0])
        rowA = firstEl + row[2:]
        matA[currentLineNum] = rowA
        if row[1] == 'M':
            rowb = firstEl + [1]
        else:
            rowb = firstEl + [-1]
        matb[currentLineNum] = rowb
        currentLineNum = currentLineNum + 1
        
        
        
    return [np.matrix(matA), np.matrix(matb)]






#matAReturn, matbReturn = read_training_data("train.data")
#print(matAReturn[55])
#print(matbReturn[55])
    
