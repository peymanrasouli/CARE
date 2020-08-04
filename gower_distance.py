import numpy as np

def GowerDistance(x1, x2, feature_range, discrete_indices, continuous_indices):
    distance = []
    if continuous_indices is not None:
        for j in continuous_indices:
            distance.append((1/feature_range[j]) * abs(x1[j]-x2[j]))

    if discrete_indices is not None:
        for j in discrete_indices:
            distance.append(int(x1[j] != x2[j]))

    return np.mean(distance)