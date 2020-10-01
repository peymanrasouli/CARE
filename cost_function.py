import numpy as np
from prediction_distance import PredictionDistance
from feature_distance import FeatureDistance
from proximity import Proximity
from sparsity import Sparsity
from actionable_recourse import ActionableRecourse
from connectedness import Connectedness
from correlation import Correlation
from mappings import theta2ord, ord2ohe, ord2org

def CostFunction(x_ord, x_theta, x_org, dataset, predict_class_fn, predict_proba_fn, discrete_indices,
                 continuous_indices, feature_width, ea_scaler, probability_thresh, cf_class, cf_range,
                 lof_model, hdbscan_model, action_operation, action_priority, corr_models, cf_theta):

    ## Constructing the counterfactual instance cf from the individual
    cf_theta = np.asarray(cf_theta)
    cf_ord = theta2ord(cf_theta, ea_scaler, discrete_indices)
    cf_ohe = ord2ohe(cf_ord, dataset)
    cf_org = ord2org(cf_ord, dataset)

    ## Objective 1: Prediction Distance
    f1 = PredictionDistance(cf_ohe, predict_class_fn, predict_proba_fn, probability_thresh, cf_class, cf_range)

    ## Objective 2: Feature Distance
    f2 = FeatureDistance(x_ord, cf_ord, feature_width, discrete_indices, continuous_indices)

    ## Objective 3: Proximity
    f3 = Proximity(cf_theta, lof_model)

    ## Objective 4: Actionable Recourse
    f4 = ActionableRecourse(x_org, cf_org, action_operation, action_priority)

    ## Objective 5: Sparsity
    f5 = Sparsity(x_org, cf_org)

    ## Objective 6: Connectedness
    f6 = Connectedness(cf_theta, hdbscan_model)

    ## Objective 7: Correlation
    f7 = Correlation(x_ord, cf_ord, cf_theta, corr_models, feature_width, discrete_indices, continuous_indices)

    return f1, f2, f3, f4, f5, f6, f7