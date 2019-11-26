from vec import Vec

def signum(u):
    '''
    input: a Vec u
    output: the Vec v with the same domain as u such that
    v[d] = +1 if u[d] >= 0 or
    v[d] = −1 if u[d] < 0
    For example, signum(Vec({'A','B'}, {'A':3, 'B':-2})) is Vec({’A’, ’B’},{’A’: 1, ’B’: -1})
    '''
    v = []
    for i in range(len(u)):
        if u[i] >= 0:
            v = v + [1]
        else:
            v = v + [-1]
    return v


print(signum([3,-2]))
