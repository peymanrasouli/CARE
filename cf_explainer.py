import numpy as np
import pandas as pd
from alibi.explainers import CounterFactual
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from evaluate_counterfactuals import EvaluateCounterfactuals
from highlight_changes import HighlightChanges

def CFExplainer(x_bb, blackbox, dataset, task, probability_thresh, MOCF_output):

    feature_names = dataset['feature_names']
    feature_encoder = dataset['feature_encoder']
    feature_scaler = dataset['feature_scaler']
    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = tuple([feature_min.reshape(1, -1), feature_max.reshape(1, -1)])

    x_bb_ = x_bb.reshape(1, -1)
    n_features = x_bb_.shape

    cf_explainer = CounterFactual(predict_fn=blackbox.predict_proba, shape=n_features,
                                  feature_range=feature_range, target_proba=probability_thresh,
                                  target_class='other')

    explanations = cf_explainer.explain(x_bb_)

    cfs = []
    cfs.append(explanations.cf['X'].ravel())
    for iter, res in explanations.all.items():
        for cf in res:
            cfs.append(cf['X'].ravel())

    cfs = np.asarray(cfs)
    cfs[:,discrete_indices] = cfs[:,discrete_indices].astype(int)
    cfs = pd.DataFrame(data=cfs, columns=feature_names)

    ## Evaluating counter-factuals
    toolbox = MOCF_output['toolbox']
    OBJ_name = MOCF_output['OBJ_name']
    ea_scaler = MOCF_output['ea_scaler']
    solutions = BB2Theta(cfs, ea_scaler)
    cfs, cfs_eval = EvaluateCounterfactuals(cfs, solutions, blackbox, toolbox, OBJ_name, task)

    ## Recovering original data
    x_original = BB2Original(x_bb, feature_encoder, feature_scaler, discrete_indices, continuous_indices)
    cfs_original = BB2Original(cfs, feature_encoder, feature_scaler, discrete_indices, continuous_indices)
    cfs_original = pd.concat([x_original, cfs_original])
    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(cfs_original) - 1)])
    cfs_original = cfs_original.set_index(index)

    ## Highlighting changes
    cfs_original_highlight = HighlightChanges(cfs_original)

    ## Returning the results
    output = {'solutions': solutions,
              'cfs': cfs,
              'cfs_original': cfs_original,
              'cfs_original_highlight': cfs_original_highlight,
              'cfs_eval': cfs_eval,
              }

    return output