import pandas as pd
from utils import *
import dice_ml
import tensorflow as tf
tf.InteractiveSession()
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train, dataset, task, MOCF_output,
                  feasibleAR=False, user_preferences=None, n_cf=5, desired_class="opposite", probability_thresh=0.5):

    # preparing ataset for DiCE model
    feature_names = dataset['feature_names']
    continuous_features = dataset['continuous_features']
    discrete_features = dataset['discrete_features']

    data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names+['class'])
    data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
    data_frame[discrete_features] = (data_frame[discrete_features]).astype(int)

    if feasibleAR is True:

        # preparing actionable recourse
        action_operation = user_preferences['action_operation']
        continuous_indices = dataset['continuous_indices']
        features_to_vary = [feature_names[i] for i, op in enumerate(action_operation) if op is not 'fix']
        permitted_range = {}
        x_org = ord2org(x_ord, dataset)
        for i, op in enumerate(action_operation):
            if (i in continuous_indices) and (type(op) is list):
                x_org_lb = x_org.copy()
                x_org_ub = x_org.copy()
                x_org_lb[i] = op[0]
                x_org_ub[i] = op[1]
                x_ord_lb = org2ord(x_org_lb, dataset)
                x_ord_ub = org2ord(x_org_ub, dataset)
                op_scaled = [x_ord_lb[i], x_ord_ub[i]]
                permitted_range[feature_names[i]] = op_scaled

        # creating data a instance
        d = dice_ml.Data(dataframe=data_frame,
                         continuous_features=continuous_features,
                         outcome_name='class',
                         permitted_range=permitted_range)

        # setting the pre-trained ML model for explainer
        backend = 'TF1'
        m = dice_ml.Model(model=blackbox, backend=backend)

        # creating a DiCE explainer instance
        exp = dice_ml.Dice(d, m)

        ## generating counter-factuals
        x_ord_dice = {}
        for key, value in zip(feature_names, list(x_ord)):
            x_ord_dice[key] = value

        for f in discrete_features:
            x_ord_dice[f] = str(int(x_ord_dice[f]))

        # Params:
        # posthoc_sparsity_param=0.1 ->[0,1]
        # posthoc_sparsity_algorithm ="linear" -> {"linear", "binary"}
        # proximity_weight=0.5,
        # diversity_weight=1.0
        # stopping_threshold=0.5
        dice_exp = exp.generate_counterfactuals(x_ord_dice, total_CFs=n_cf, desired_class=desired_class,
                                                stopping_threshold=probability_thresh,
                                                posthoc_sparsity_algorithm="binary",
                                                features_to_vary=features_to_vary)
    else:
        # creating data a instance
        d = dice_ml.Data(dataframe=data_frame,
                         continuous_features=continuous_features,
                         outcome_name='class')

        # setting the pre-trained ML model for explainer
        backend = 'TF1'
        m = dice_ml.Model(model=blackbox, backend=backend)

        # creating a DiCE explainer instance
        exp = dice_ml.Dice(d, m)

        ## generating counter-factuals
        x_ord_dice = {}
        for key, value in zip(feature_names, list(x_ord)):
            x_ord_dice[key] = value

        for f in discrete_features:
            x_ord_dice[f] = str(int(x_ord_dice[f]))

        # Params:
        # posthoc_sparsity_param=0.1 ->[0,1]
        # posthoc_sparsity_algorithm ="linear" -> {"linear", "binary"}
        # proximity_weight=0.5,
        # diversity_weight=1.0
        # stopping_threshold=0.5
        dice_exp = exp.generate_counterfactuals(x_ord_dice, total_CFs=n_cf, desired_class=desired_class,
                                                stopping_threshold=probability_thresh,
                                                posthoc_sparsity_algorithm="binary")

    ## extracting solutions
    cfs_ord = dice_exp.final_cfs_df.iloc[:,:-1]
    cfs_ord[discrete_features] = cfs_ord[discrete_features].astype(int)

    ## evaluating counter-factuals
    toolbox = MOCF_output['toolbox']
    objective_names = MOCF_output['objective_names']
    objective_weights = MOCF_output['objective_weights']
    featureScaler = MOCF_output['featureScaler']
    feature_names = dataset['feature_names']

    cfs_ord, cfs_eval, x_cfs_ord, x_cfs_eval = evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn,
                                                                       predict_proba_fn, task, toolbox,
                                                                       objective_names, objective_weights,
                                                                       featureScaler, feature_names)

    # recovering counter-factuals in original format
    x_org, cfs_org, x_cfs_org, x_cfs_highlight = recoverOriginals(x_ord, cfs_ord, dataset, feature_names)

    # returning the results
    output = {'cfs_ord': cfs_ord,
              'cfs_org': cfs_org,
              'cfs_eval': cfs_eval,
              'x_cfs_ord': x_cfs_ord,
              'x_cfs_eval': x_cfs_eval,
              'x_cfs_org': x_cfs_org,
              'x_cfs_highlight': x_cfs_highlight,
              'toolbox': toolbox,
              'featureScaler': featureScaler,
              'objective_names': objective_names,
              'objective_weights': objective_weights,
              }

    return output