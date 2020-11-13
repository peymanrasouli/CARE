def sparsity(x_org, cf_org):
    cost = sum(x_org != cf_org)
    return cost
