import numpy as np
def Proximity(theta_cf, nbrs_gt, theta_gt):
    ## Finding closet correctly predicted instance to counterfactual (a0)
    dist_cf_a0, ind_a0 = nbrs_gt.kneighbors(theta_cf.reshape(1,-1))
    dist_cf_a0 = dist_cf_a0[0,0]
    ind_a0 = ind_a0[0,0]
    a0 = theta_gt[ind_a0]

    ## Finding the minimum distance of a0 and the rest of the correctly predicted instances
    dist_a0_xi, ind_xi = nbrs_gt.kneighbors(a0.reshape(1, -1))
    dist_a0_xi = dist_a0_xi[0,1]

    ## Calculating the proximity value
    distance = (dist_cf_a0 / dist_a0_xi)
    distance = 99999 if distance == 0.0 or distance > 1.0 else distance
    return distance
