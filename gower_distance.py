import numpy as np

def GowerDistance(x1, x2, var_min, var_max, disc_ind, cont_ind):
    distance = []
    if cont_ind is not None:
        for j in cont_ind:
            Rj = var_max[j] - var_min[j]
            distance.append((1/Rj) * abs(x1[j]-x2[j]))

    if disc_ind is not None:
        for j in disc_ind:
            distance.append(int(x1[j] != x2[j]))

    return np.mean(distance)