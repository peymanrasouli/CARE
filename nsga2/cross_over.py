import numpy as np

def Crossover(x1,x2,co):

    prob = np.random.rand(len(x1))
    y1 = np.zeros(np.shape(x1))
    y2 = np.zeros(np.shape(x1))
    mask1 = prob < co
    mask2 = np.invert(mask1)
    y1[mask1] = x1[mask1]
    y1[mask2] = x2[mask2]
    y2[mask1] = x2[mask1]
    y2[mask2] = x1[mask2]

    return y1, y2