def Sparsity(theta_x, theta_cf):
    return sum(theta_x != theta_cf)