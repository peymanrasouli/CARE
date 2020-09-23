def Sparsity(x_theta, cf_theta):
    return sum(x_theta != cf_theta)
