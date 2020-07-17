import numpy as np
import math
import random

def Mutate(x,mu,VarMin,VarMax):
    nVar = np.size(x)

    nmu = math.ceil(mu*nVar)

    J = random.sample(range(1,nVar),nmu)

    sigma = 0.1 * (VarMax - VarMin)

    shape_x = np.shape(x)
    x = x.reshape(-1)
    sigma = sigma.reshape(-1)

    y = x.copy()
    for j in J:
        y[j] = x[j] + sigma[j] * np.random.randn(1)

    y = y.reshape(shape_x)
    y = np.maximum(y,VarMin)
    y = np.minimum(y,VarMax)

    return y
