import pandas as pd
from utils import *
import dice_ml
import tensorflow as tf
tf.InteractiveSession()
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train, dataset, task, CARE_output,
                  explainer=None, ACTIONABILITY=False, user_preferences=None, n_cf=5, desired_class="opposite",
                  probability_thresh=0.5, proximity_weight=0.5, diversity_weight=1.0, features_to_vary='all'):

    # reading the data set information
    feature_names = dataset['feature_names']
    continuous_features = dataset['continuous_features']
    discrete_features = dataset['discrete_features']

    # creating an explainer instance in case it is not pre-created
    if explainer is None:

        # preparing dataset for DiCE model
        data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names+['class'])
        data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
        data_frame[discrete_features] = (data_frame[discrete_features]).astype(int)

        # setting actions to explainer in case actionable recourse is applied
        if ACTIONABILITY is True:

            # preparing actions for DiCE
            constraint = user_preferences['constraint']
            continuous_indices = dataset['continuous_indices']
            features_to_vary = [feature_names[i] for i, c in enumerate(constraint) if c is not 'fix']
            permitted_range = {}
            x_org = ord2org(x_ord, dataset)
            for i, c in enumerate(constraint):
                if (i in continuous_indices) and (type(c) is list):
                    x_org_lb = x_org.copy()
                    x_org_ub = x_org.copy()
                    x_org_lb[i] = c[0]
                    x_org_ub[i] = c[1]
                    x_ord_lb = org2ord(x_org_lb, dataset)
                    x_ord_ub = org2ord(x_org_ub, dataset)
                    range_scaled = [x_ord_lb[i], x_ord_ub[i]]
                    permitted_range[feature_names[i]] = range_scaled
                if (i in continuous_indices) and (c == 'ge'):
                    permitted_range[feature_names[i]] = [x_ord[i], dataset['feature_ranges'][feature_names[i]][1]]
                if (i in continuous_indices) and (c == 'le'):
                    permitted_range[feature_names[i]] = [dataset['feature_ranges'][feature_names[i]][0], x_ord[i]]

            # creating data a instance
            data = dice_ml.Data(dataframe=data_frame,
                             continuous_features=continuous_features,
                             outcome_name='class',
                             permitted_range=permitted_range)

            # setting the pre-trained ML model for explainer
            backend = 'TF1'
            model = dice_ml.Model(model=blackbox, backend=backend)

            # creating a DiCE explainer instance
            explainer = dice_ml.Dice(data, model)

        else:

            # creating data a instance
            data = dice_ml.Data(dataframe=data_frame,
                             continuous_features=continuous_features,
                             outcome_name='class')

            # setting the pre-trained ML model for explainer
            backend = 'TF1'
            model = dice_ml.Model(model=blackbox, backend=backend)

            # creating a DiCE explainer instance
            explainer = dice_ml.Dice(data, model)

    # preparing instance to explain for DiCE
    x_ord_dice = {}
    for key, value in zip(feature_names, list(x_ord)):
        x_ord_dice[key] = value

    for f in discrete_features:
        x_ord_dice[f] = str(int(x_ord_dice[f]))

    # generating counterfactuals
    explanations = explainer.generate_counterfactuals(x_ord_dice, total_CFs=n_cf,
                                                      desired_class=desired_class,
                                                      stopping_threshold=probability_thresh,
                                                      posthoc_sparsity_algorithm="binary",
                                                      features_to_vary=features_to_vary,
                                                      proximity_weight=proximity_weight,
                                                      diversity_weight=diversity_weight)

    # extracting solutions
    cfs_ord = explanations.final_cfs_df.iloc[:,:-1]
    cfs_ord[discrete_features] = cfs_ord[discrete_features].astype(int)

    # evaluating counterfactuals
    toolbox = CARE_output['toolbox']
    objective_names = CARE_output['objective_names']
    featureScaler = CARE_output['featureScaler']

    cfs_ord, \
    cfs_eval, \
    x_cfs_ord, \
    x_cfs_eval = evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                                         toolbox, objective_names, featureScaler, feature_names)

    # recovering counterfactuals in original format
    x_org, \
    cfs_org, \
    x_cfs_org, \
    x_cfs_highlight = recoverOriginals(x_ord, cfs_ord, dataset, feature_names)

    # best counterfactual
    best_cf_ord = cfs_ord.iloc[0]
    best_cf_org = cfs_org.iloc[0]
    best_cf_eval = cfs_eval.iloc[0]

    # returning the results
    output = {'cfs_ord': cfs_ord,
              'cfs_org': cfs_org,
              'cfs_eval': cfs_eval,
              'best_cf_ord': best_cf_ord,
              'best_cf_org': best_cf_org,
              'best_cf_eval': best_cf_eval,
              'x_cfs_ord': x_cfs_ord,
              'x_cfs_eval': x_cfs_eval,
              'x_cfs_org': x_cfs_org,
              'x_cfs_highlight': x_cfs_highlight,
              }

    return output