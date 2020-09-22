import numpy as np
from alibi.explainers import CounterFactual

def CFExplainer(x, blackbox, dataset, probability_thresh):

    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = tuple([feature_min.reshape(1, -1), feature_max.reshape(1, -1)])

    x = x.reshape(1, -1)
    n_features = x.shape

    cf_explainer = CounterFactual(predict_fn=blackbox.predict_proba, shape=n_features,
                                  feature_range=feature_range, target_proba=probability_thresh)

    explanations = cf_explainer.explain(x)


    print('')