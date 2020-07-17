import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances

def CostFunction(cf, blackbox, MAD, x):

    cf = np.round(cf)

    # z1 = pairwise_distances(cf.reshape(1,-1),x.reshape(1,-1)).ravel()[0]
    z1 = np.sum(np.abs(cf - x) / MAD)


    label_x =  blackbox.predict(x.reshape(1,-1))
    pproba_cf = blackbox.predict_proba(cf.reshape(1, -1))

    # z2 = 1  - pproba_cf[0, int(not(label_x))]    ## Counter factual samples
    z2 = np.abs(0.5 - pproba_cf[0, label_x])       ## Flip point samples

    z = [z1,z2]

    sol= {
        'distance_:': z1,
        'pproba_cf':pproba_cf
    }

    return z , sol

# def manhattan_distance(X, X_hat):
#
#             nonzero = np.where(MAD > 0)
#             sim_idx = list()
#             for i in range(X_hat.shape[0]):
#                 diff = np.abs(X - X_hat[i]) / MAD
#                 diff_nonzero = np.squeeze(diff[:, nonzero])
#                 sum_dist = np.sum(diff_nonzero, axis=1)
#                 sim_idx.append(np.argmin(sum_dist))
#             return sim_idx