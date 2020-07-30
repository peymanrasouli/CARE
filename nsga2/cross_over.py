import numpy as np

def Crossover(x1,x2,gamma,VarMin,VarMax):

    alpha = np.random.uniform(-gamma,1+gamma,x1.shape)

    y1 = alpha*x1 + (1 - alpha)*x2
    y2 = alpha*x2 + (1 - alpha)*x1

    y1 = np.maximum(y1, VarMin)
    y1 = np.minimum(y1, VarMax)

    y2 = np.maximum(y2, VarMin)
    y2 = np.minimum(y2, VarMax)

    return y1, y2