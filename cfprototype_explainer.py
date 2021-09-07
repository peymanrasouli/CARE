import pandas as pd
from utils import *
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset, task,
                         CARE_output, explainer=None, target_class=None, n_cf=5):

    # preparing instance to explain for CFPrototype
    x_ohe = ord2ohe(x_ord,dataset)
    x_ohe = x_ohe.reshape((1,) + x_ohe.shape)

    # creating an explainer instance in case it is not pre-created
    if explainer is None:

        # preparing parameters
        cat_vars_ord = {}
        for i, d in enumerate(dataset['discrete_indices']):
            cat_vars_ord[d] = dataset['n_cat_discrete'][i]
        cat_vars_ohe = ord_to_ohe(X_train, cat_vars_ord)[1]
        ohe = True if dataset['discrete_availability'] else False
        shape = x_ohe.shape
        rng_min = np.min(X_train, axis=0)
        rng_max = np.max(X_train, axis=0)
        rng = tuple([rng_min.reshape(1, -1), rng_max.reshape(1, -1)])
        rng_shape = (1,) + X_train.shape[1:]
        feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32),
                         (np.ones(rng_shape) * rng[1]).astype(np.float32))

        # creating prototype counterfactual explainer
        explainer = CounterFactualProto(predict=predict_proba_fn, shape=shape, feature_range=feature_range,
                                        cat_vars=cat_vars_ohe, ohe=ohe, beta=0.1, theta=10,
                                        use_kdtree=True, max_iterations=500, c_init=1.0, c_steps=5)

        # Fitting the explainer on the training data
        X_train_ohe = ord2ohe(X_train, dataset)
        explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

    # generating counterfactuals
    explanations = explainer.explain(x_ohe,target_class=target_class)

    # extracting solutions
    cfs_iter = []
    for iter, res in explanations.all.items():
       cfs_iter.append(res)

    cfs = []
    cfs.append(explanations.cf['X'].ravel())
    for item in reversed(cfs_iter):
        for cf in item:
            cfs.append(cf.ravel())

    cfs_ohe = np.asarray(cfs)
    n_cf_ = min(n_cf, cfs_ohe.shape[0])
    cfs_ohe = cfs_ohe[:n_cf_,:]
    cfs_ord = ohe2ord(cfs_ohe, dataset)
    cfs_ord = pd.DataFrame(data=cfs_ord, columns=dataset['feature_names'])

    # evaluating counterfactuals
    toolbox = CARE_output['toolbox']
    objective_names = CARE_output['objective_names']
    featureScaler = CARE_output['featureScaler']
    feature_names = dataset['feature_names']

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