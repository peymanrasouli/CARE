def Sparsity(x_bb, cf_bb):
    cost = sum(x_bb != cf_bb)
    return cost
