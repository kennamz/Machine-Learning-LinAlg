import numpy as np

def fraction_wrong(A, b, w):
    # for r in number of rows of A
    #   if sign(r*w) = sign(b(i))
    #       signMatches++
    signMatches = 0
    for i in range(len(A)):
        rowA = np.matrix(A[i])
        product = rowA.dot(w)
        if (product > 0 and b[i] > 0) or (product < 0 and b[i] < 0):
            signMatches = signMatches + 1
    return 1-(signMatches/len(A))
