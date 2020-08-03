import numpy as np
import math
import random

def Mutate(x, mu, var_min, var_max, disc_ind, cont_ind):

    nVar = np.size(x)

    nmu = math.ceil(mu*nVar)

    J = random.sample(range(1,nVar),nmu)

    sigma = 0.1 * (var_max - var_min)

    y = x.copy()
    for j in J:
        if cont_ind is not None:
            if j in cont_ind:
                y[j] = x[j] + sigma[j] * np.random.randn(1)
        if disc_ind is not None:
            if j in disc_ind:
                y[j] = np.random.choice(range(var_min[j], var_max[j]))

    y = np.maximum(y,var_min)
    y = np.minimum(y,var_max)

    return y
