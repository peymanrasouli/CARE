import numpy as np

def Sparsity(x_bb, cf_bb):
    diff = sum(x_bb != cf_bb)
    cost = max(np.random.rand(), diff)
    return cost
