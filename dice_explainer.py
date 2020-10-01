import numpy as np
import pandas as pd
import dice_ml
import tensorflow as tf
tf.InteractiveSession()
from evaluate_counterfactuals import EvaluateCounterfactuals
from recover_originals import RecoverOriginals

def DiCEExplainer(x_ord, blackbox, predict_class_fn, predict_proba_fn, X_train, Y_train,
                  dataset, task, MOCF_output, n_cf=5, probability_thresh=0.5):

    ## Dataset for DiCE model
    feature_names = dataset['feature_names']
    continuous_features = dataset['continuous_features']
    discrete_features = dataset['discrete_features']

    data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names+['class'])
    data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
    data_frame[discrete_features] = (data_frame[discrete_features]).astype(int)

    d = dice_ml.Data(dataframe=data_frame,
                     continuous_features=continuous_features,
                     outcome_name='class')

    ## Pre-trained ML model
    backend = 'TF' + tf.__version__[0]  # TF1
    m = dice_ml.Model(model=blackbox, backend=backend)

    # DiCE explanation instance
    exp = dice_ml.Dice(d, m)

    ## Generating explanations for x_ord
    x_ord_dice = {}
    for key,value in zip(feature_names,list(x_ord)):
        x_ord_dice[key] = value

    for f in discrete_features:
        x_ord_dice[f] = str(int(x_ord_dice[f]))

    dice_exp = exp.generate_counterfactuals(x_ord_dice, total_CFs=n_cf, stopping_threshold=probability_thresh)

    ## Extracting solutions
    cfs_ord = dice_exp.final_cfs_df.iloc[:,:-1]
    cfs_ord[discrete_features] = cfs_ord[discrete_features].astype(int)

    ## Evaluating counter-factuals
    cfs_ord, cfs_eval = EvaluateCounterfactuals(cfs_ord, dataset, predict_class_fn, predict_proba_fn, task, MOCF_output)

    ## Recovering original data
    x_org, cfs_org, x_cfs_org, x_cfs_highlight = RecoverOriginals(x_ord, cfs_ord, dataset)

    ## Returning the results
    output = {'cfs_ord': cfs_ord,
              'cfs_org': cfs_org,
              'cfs_eval': cfs_eval,
              'x_cfs_org': x_cfs_org,
              'x_cfs_highlight': x_cfs_highlight,
              }

    return output