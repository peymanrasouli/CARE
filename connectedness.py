import numpy as np
from sklearn.cluster import DBSCAN

def Connectedness(theta_cf, nbrs_gt, theta_gt):

    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1, -1))
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]
    epsilon =  dist_a0_xi + 0.001

    ## Clustering the potential counterfactual along with correctly classified instances
    theta_gt = theta_gt[ind_xi[0]]
    theta = np.r_[theta_gt, theta_cf.reshape(1, -1)]
    clustering = DBSCAN(eps=epsilon, min_samples=2, metric='minkowski', p=2).fit(theta)

    ## Calculating eps-chain
    if clustering.labels_[0] == clustering.labels_[-1]:
        chain = len(np.where(clustering.labels_ == clustering.labels_[0])[0])
    else:
        chain = 0

    return chain