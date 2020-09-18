import numpy as np

def FeatureDistance(x, cf, feature_width, discrete_indices, continuous_indices):
    distance = []
    if continuous_indices is not None:
        for j in continuous_indices:
            distance.append((1/feature_width[j]) * abs(x[j]-cf[j]))
    if discrete_indices is not None:
        for j in discrete_indices:
            distance.append(int(x[j] != cf[j]))
    return np.mean(distance)