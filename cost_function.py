import numpy as np
from gower_distance import GowerDistance
from prediction_distance import PredictionDistance
from proximity import Proximity
from connectedness import Connectedness
# from proximity_connectedness import ProximityConnectedness
from sparsity import Sparsity

def CostFunction(x, theta_x, discrete_indices, continuous_indices,
                 mapping_scale, mapping_offset, feature_range, blackbox,
                 probability_range, response_range, cf_label,
                 nbrs_gt, theta_gt, epsilon, theta_cf):

    ## Constructing the counterfactual instance
    theta_cf = np.asarray(theta_cf)
    cf = theta_cf * mapping_scale + mapping_offset
    cf[discrete_indices] = np.rint(cf[discrete_indices])

    ## Objective 1: opposite outcome
    f1 = PredictionDistance(cf, blackbox, probability_range, response_range, cf_label)

    ## Objective 2: distnace
    f2 = GowerDistance(x, cf, feature_range, discrete_indices, continuous_indices)

    ## Objective 3: proximity
    f3 = Proximity(theta_cf, nbrs_gt, theta_gt)
    # f3 = ProximityConnectedness(theta_cf, nbrs_gt, theta_gt)

    ## Objective 4: actionable
    # f4 = 0

    ## Objective 5: sparsity
    f5 = Sparsity(x, cf, feature_range, discrete_indices, continuous_indices, crisp_thresh=0.05)

    ## Objective 6: connectedness
    f6 = Connectedness(theta_cf, nbrs_gt, theta_gt)

    return f1, f2, f3, f5, f6