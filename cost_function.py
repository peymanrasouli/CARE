import numpy as np
from gower_distance import GowerDistance
from prediction_distance import PredictionDistance
from sparsity import Sparsity

def CostFunction(x, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
                 feature_range, blackbox, probability_range, response_range, cf_label, p):

    ## Constructing the counterfactual instance
    cf = p * mapping_scale + mapping_offset
    cf[discrete_indices] = np.rint(cf[discrete_indices])

    ## Objective 1: opposite outcome
    f1 = PredictionDistance(cf, blackbox, probability_range, response_range, cf_label)


    ## Objective 2: proximity
    f2 = GowerDistance(x, cf, feature_range, discrete_indices, continuous_indices)

    ## Objective 3: connectedness
    f3 = 0


    ## Objective 4: actionable
    f4 = 0


    ## Objective 5: sparsity
    f5 = Sparsity(x, cf, feature_range, discrete_indices, continuous_indices, crisp_thresh=0.1)



    return f1, f2, f5