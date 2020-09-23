import numpy as np
from prediction_distance import PredictionDistance
from feature_distance import FeatureDistance
from proximity import Proximity
from sparsity import Sparsity
from actionable_recourse import ActionableRecourse
from connectedness import Connectedness
from correlation import Correlation

def CostFunction(x_bb, x_theta, discrete_indices, continuous_indices, feature_encoder, feature_scaler, ea_scaler,
                 feature_width, blackbox, probability_thresh, cf_label, cf_range, lof_model, hdbscan_model,
                 action_operation, action_priority, corr_models, cf_theta):

    ## Constructing the counterfactual instance cf from the individual
    cf_theta = np.asarray(cf_theta)
    cf_bb = ea_scaler.inverse_transform(cf_theta.reshape(1, -1)).ravel()
    cf_bb[discrete_indices] = np.rint(cf_bb[discrete_indices])

    ## Objective 1: Prediction Distance
    f1 = PredictionDistance(cf_bb, blackbox, probability_thresh, cf_label, cf_range)

    ## Objective 2: Feature Distance
    f2 = FeatureDistance(x_bb, cf_bb, feature_width, discrete_indices, continuous_indices)

    ## Objective 3: Proximity
    f3 = Proximity(cf_theta, lof_model)

    ## Objective 4: Actionable Recourse
    f4 = ActionableRecourse(x_bb, cf_bb, action_operation, action_priority)

    ## Objective 5: Sparsity
    f5 = Sparsity(x_theta, cf_theta)

    ## Objective 6: Connectedness
    f6 = Connectedness(cf_theta, hdbscan_model)

    ## Objective 7: Correlation
    f7 = Correlation(x_bb, cf_bb, cf_theta, corr_models, feature_width, discrete_indices, continuous_indices)

    return f1, f2, f3, f4, f5, f6, f7