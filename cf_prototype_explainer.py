import numpy as np
import pandas as pd
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ohe_to_ord, ord_to_ohe
from mappings import ord2ohe, ohe2ord
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def CFPrototypeExplainer(x_ord, predict_class_fn, predict_proba_fn, X_train, dataset, task, MOCF_output):
    ## Preparing parameters
    cat_vars_ord = {}
    for i,d in enumerate(dataset['discrete_indices']):
        cat_vars_ord[d] = dataset['n_cat_discrete'][i]
    cat_vars_ohe = ord_to_ohe(X_train, cat_vars_ord)[1]

    x_ohe = ord2ohe(x_ord,dataset)
    x_ohe = x_ohe.reshape((1,) + x_ohe.shape)
    shape = x_ohe.shape
    beta = .01
    c_init = 1.
    c_steps = 5
    max_iterations = 500
    rng_min = np.min(X_train, axis=0)
    rng_max = np.max(X_train, axis=0)
    rng = tuple([rng_min.reshape(1, -1), rng_max.reshape(1, -1)])
    rng_shape = (1,) + X_train.shape[1:]
    feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32),
                     (np.ones(rng_shape) * rng[1]).astype(np.float32))

    ## Creating prototype counter-factual explainer
    prototype_cf_explainer = CounterFactualProto(predict=predict_proba_fn, shape=shape, beta=beta,
                                                 feature_range=feature_range, cat_vars=cat_vars_ohe, ohe=True,
                                                 c_init=c_init, c_steps=c_steps, max_iterations = max_iterations)

    ## Fitting the explainer on the training data
    X_train_ohe = ord2ohe(X_train, dataset)
    prototype_cf_explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

    ## Generating counter-factuals
    explanations = prototype_cf_explainer.explain(x_ohe)

    ## Extracting solutions
    cfs = []
    cfs.append(explanations.cf['X'].ravel())
    for iter, res in explanations.all.items():
        for cf in res:
            cfs.append(cf.ravel())
    feature_names = dataset['feature_names']

    cfs_ohe = np.asarray(cfs)
    cfs_ord = ohe2ord(cfs_ohe, dataset)
    cfs_ord = pd.DataFrame(data=cfs_ord, columns=feature_names)

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