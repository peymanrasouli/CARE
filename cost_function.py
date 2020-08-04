import numpy as np
from gower_distance import GowerDistance
from prediction_distance import PredictionDistance

def CostFunction(x, l_cf, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
                 feature_range, blackbox, probability_range, p):

    cf = p * mapping_scale + mapping_offset
    cf[discrete_indices] = np.rint(cf[discrete_indices])

    ## Objective 1: opposite outcome
    f1 = 0


    ## Objective 2: proximity
    f2 = 0


    ## Objective 3: connectedness
    f3 = 0


    ## Objective 4: actionable
    f4 = 0


    ## Objective 5: sparsity
    f5 = 0


    f1 = PredictionDistance(cf, l_cf, blackbox, probability_range)

    f2 = GowerDistance(x, cf, feature_range, discrete_indices, continuous_indices)


    return f1 , f2