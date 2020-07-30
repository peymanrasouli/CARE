import numpy as np

def Dominates(p,q):
    p = np.asarray([i for i in p['cost']])
    q = np.asarray([i for i in q['cost']])
    b = all(p <= q) and any(p < q)
    return b