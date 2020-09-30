import numpy as np
from prediction_distance import PredictionDistance
from feature_distance import FeatureDistance
from proximity import Proximity
from sparsity import Sparsity
from actionable_recourse import ActionableRecourse
from connectedness import Connectedness
from correlation import Correlation

def CostFunction(x_ord, x_theta, x_original, blackbox, predict_class_fn, predict_proba_fn, discrete_indices,
                 continuous_indices, ea_scaler, probability_thresh, cf_label, cf_range,lof_model,
                 hdbscan_model, action_operation, action_priority, corr_models, cf_theta):

    ## Constructing the counterfactual instance cf from the individual
    cf_theta = np.asarray(cf_theta)
    cf_bb = Theta2BB(cf_theta, ea_scaler)
    cf_original = BB2Original(cf_bb, feature_encoder, feature_scaler, discrete_indices, continuous_indices)

    ## Objective 1: Prediction Distance
    f1 = PredictionDistance(cf_bb, blackbox, probability_thresh, cf_label, cf_range)

    ## Objective 2: Feature Distance
    f2 = FeatureDistance(x_bb, cf_bb, feature_width, discrete_indices, continuous_indices)

    ## Objective 3: Proximity
    f3 = Proximity(cf_theta, lof_model)

    ## Objective 4: Actionable Recourse
    f4 = ActionableRecourse(x_original, cf_original, action_operation, action_priority)

    ## Objective 5: Sparsity
    f5 = Sparsity(x_bb, cf_bb)

    ## Objective 6: Connectedness
    f6 = Connectedness(cf_theta, hdbscan_model)

    ## Objective 7: Correlation
    f7 = Correlation(x_bb, cf_bb, cf_theta, corr_models, feature_width, discrete_indices, continuous_indices)

    return f1, f2, f3, f4, f5, f6, f7