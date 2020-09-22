import numpy as np
from alibi.explainers import CounterFactualProto
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def CFPrototypeExplainer(x, blackbox, dataset, X_train):

    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = tuple([feature_min.reshape(1, -1), feature_max.reshape(1, -1)])

    x = x.reshape(1, -1)
    n_features = x.shape

    cat_vars = {}
    for d in dataset['discrete_indices']:
        cat_vars[d] = int(feature_range[1][0,d] + 1)

    # # Prototype CF
    prototype_cf_explainer = CounterFactualProto(predict=blackbox.predict_proba, shape=n_features,
                                                 feature_range=feature_range, cat_vars=cat_vars, ohe=False,)

    prototype_cf_explainer.fit(X_train, d_type='abdm', disc_perc=[25, 50, 75])

    explanations = prototype_cf_explainer.explain(x)

    print('')