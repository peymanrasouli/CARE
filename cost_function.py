import numpy as np
from prediction_distance import PredictionDistance
from gower_distance import GowerDistance
from proximity import Proximity
from sparsity import Sparsity
from connectedness import Connectedness

def CostFunction(x, theta_x, discrete_indices, continuous_indices,
                 mapping_scale, mapping_offset, feature_range, blackbox,
                 probability_range, response_range, cf_label, lof_model,
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
    f3 = Proximity(theta_cf, lof_model)

    ## Objective 4: actionable
    # f4 = 0

    ## Objective 5: sparsity
    f5 = Sparsity(theta_x, theta_cf)

    ## Objective 6: connectedness
    # f6 = Connectedness(theta_cf, nbrs_gt, theta_gt)

    return f1, f2, f3, f5