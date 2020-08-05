import numpy as np

def Sparsity(x1, x2, feature_range, discrete_indices, continuous_indices, continuous_thresh=0.1):
    changed = []
    if continuous_indices is not None:
        for j in continuous_indices:
            changed.append((1/feature_range[j]) * abs(x1[j]-x2[j]))

    if discrete_indices is not None:
        for j in discrete_indices:
            changed.append(int(x1[j] != x2[j]))

    changed = np.asarray(changed)
    changed[np.where(changed > continuous_thresh)] = 1
    changed[np.where(changed < continuous_thresh)] = 0
    return sum(changed)