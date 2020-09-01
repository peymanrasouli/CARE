import numpy as np
from prediction_distance import PredictionDistance
from feature_distance import FeatureDistance
from proximity import Proximity
from sparsity import Sparsity
from actionable_recourse import ActionableRecourse
from connectedness import Connectedness

def CostFunction(x, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
                 feature_range, blackbox, probability_thresh, response_range, cf_label,
                 lof_model, hdbscan_model, actions_o, actions_w, theta_cf):

    ## Constructing the counterfactual instance
    theta_cf = np.asarray(theta_cf)
    cf = (theta_cf * mapping_scale + mapping_offset).astype(int)

    ## Objective 1: Prediction Distance
    f1 = PredictionDistance(cf, blackbox, probability_thresh, response_range, cf_label)

    ## Objective 2: Feature Distance
    f2 = FeatureDistance(x, cf, feature_range, discrete_indices, continuous_indices)

    ## Objective 3: Proximity
    f3 = Proximity(theta_cf, lof_model)

    ## Objective 4: Actionable Recourse
    f4 = ActionableRecourse(x, cf, actions_o, actions_w)

    ## Objective 5: Sparsity
    f5 = Sparsity(x, cf)

    ## Objective 6: Connectedness
    f6 = Connectedness(theta_cf, hdbscan_model)

    return f1, f2, f3, f4, f5, f6