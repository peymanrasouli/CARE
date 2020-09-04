import numpy as np
from sklearn.preprocessing import MinMaxScaler

def Correlation(x, cf, feature_range, discrete_indices, continuous_indices, corr):

    diff = cf - x
    diff_degree = np.zeros(diff.shape)
    for i in discrete_indices:
        diff_degree[i] = 0 if diff[i] == 0 else 1
    for j in continuous_indices:
        diff_degree[j] = 0 if diff[j] == 0 else diff[j]/feature_range[j]

    change_ratio = np.zeros(corr.shape)
    for i in range(len(diff_degree)):
        for j in range(len(diff_degree)):
            change_ratio[i, j] = diff_degree[i] / diff_degree[j]
            change_ratio[j, i] = diff_degree[i] / diff_degree[j]

    change_ratio[np.isinf(change_ratio)] = 0
    change_ratio[np.isnan(change_ratio)] = 0

    change_ratio[:,discrete_indices] = np.abs(change_ratio[:,discrete_indices])
    change_ratio[discrete_indices,:] = np.abs(change_ratio[discrete_indices,:])


    return 0
    # diff = np.zeros(corr.shape)


    # d[discrete_indices] = np.abs(d[discrete_indices])
    #
    # for i in range(len(d)):
    #     for j in range(len(d)):
    #         diff[i,j] = d[i] * d[j]
    #
    # nonzero_d = np.nonzero(d)
    # for i in nonzero_d:
    #     for j in nonzero_d:
    #         diff[i,j] = 0
    #
    # prod = corr * diff
    # r = np.sum(np.sum(prod)) / np.count_nonzero(diff)

    # return r

    # diff[np.isinf(diff)] = 0
    #
    # prod = corr * diff


    # for i in range(diff.shape[0]):
    #     d1 = (theta_cf[i] - theta_x[i]) / 2
    #     a = float(bool(d1)) if i in discrete_indices else d1
    #
    #     for j in range(i,diff.shape[1]):
    #         d2 = (theta_cf[j] - theta_x[j]) / 2
    #         b = float(bool(d2)) if j in discrete_indices else d2
    #
    #         # diff[i, j] = 0 if max([a,b]) else min([a,b])/max([a,b])
    #         diff[i, j] = 0 if a == b == 0 else b/a
    #         diff[j, i] = 0 if a == b == 0 else b/a
    #
    # diff[np.isinf(diff)] = 0
    #
    # prod = corr * diff
    # r = np.sum(np.sum(prod))/ np.count_nonzero(diff)

    # return r