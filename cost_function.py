import numpy as np
from prediction_distance import PredictionDistance
from feature_distance import FeatureDistance
from proximity import Proximity
from sparsity import Sparsity
from actionable_recourse import ActionableRecourse
from connectedness import Connectedness
from correlation import Correlation

def CostFunction(x, theta_x, discrete_indices, continuous_indices, mapping_scale, mapping_offset,
                 feature_range, blackbox, probability_thresh, cf_label, cf_range,
                 lof_model, hdbscan_model, actions, corr, theta_cf):

    ## Constructing the counterfactual instance
    theta_cf = np.asarray(theta_cf)
    cf = np.rint(theta_cf * mapping_scale + mapping_offset)

    ## Objective 1: Prediction Distance
    f1 = PredictionDistance(cf, blackbox, probability_thresh, cf_label, cf_range)

    ## Objective 2: Feature Distance
    f2 = FeatureDistance(x, cf, feature_range, discrete_indices, continuous_indices)

    ## Objective 3: Proximity
    f3 = Proximity(theta_cf, lof_model)

    ## Objective 4: Actionable Recourse
    f4 = ActionableRecourse(x, cf, actions)

    ## Objective 5: Sparsity
    f5 = Sparsity(x, cf)

    ## Objective 6: Connectedness
    f6 = Connectedness(theta_cf, hdbscan_model)

    ## Objective 6: Connectedness
    f7 = Correlation(x, cf, feature_range, discrete_indices, continuous_indices, corr)

    return f1, f2, f3, f4, f5, f6, f7