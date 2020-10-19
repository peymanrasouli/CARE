import pandas as pd
from utils import *
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset, task, MOCF_output, target_class=None):
    # preparing parameters
    cat_vars_ord = {}
    for i,d in enumerate(dataset['discrete_indices']):
        cat_vars_ord[d] = dataset['n_cat_discrete'][i]
    cat_vars_ohe = ord_to_ohe(X_train, cat_vars_ord)[1]

    x_ohe = ord2ohe(x_ord,dataset)
    x_ohe = x_ohe.reshape((1,) + x_ohe.shape)
    shape = x_ohe.shape
    rng_min = np.min(X_train, axis=0)
    rng_max = np.max(X_train, axis=0)
    rng = tuple([rng_min.reshape(1, -1), rng_max.reshape(1, -1)])
    rng_shape = (1,) + X_train.shape[1:]
    feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32),
                     (np.ones(rng_shape) * rng[1]).astype(np.float32))

    # creating prototype counter-factual explainer
    cfprototype_explainer = CounterFactualProto(predict=predict_proba_fn, shape=shape, feature_range=feature_range,
                                                cat_vars=cat_vars_ohe, ohe=True, beta=0.1, theta=10,
                                                use_kdtree=True, max_iterations=500, c_init=1.0, c_steps=5)

    # Fitting the explainer on the training data
    X_train_ohe = ord2ohe(X_train, dataset)
    cfprototype_explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

    # generating counter-factuals
    explanations = cfprototype_explainer.explain(x_ohe,target_class=target_class)

    # extracting solutions
    cfs = []
    cfs.append(explanations.cf['X'].ravel())
    for iter, res in explanations.all.items():
        for cf in res:
            cfs.append(cf.ravel())
    feature_names = dataset['feature_names']

    cfs_ohe = np.asarray(cfs)
    cfs_ord = ohe2ord(cfs_ohe, dataset)
    cfs_ord = pd.DataFrame(data=cfs_ord, columns=feature_names)

    # evaluating counter-factuals
    toolbox = MOCF_output['toolbox']
    objective_names = MOCF_output['objective_names']
    featureScaler = MOCF_output['featureScaler']
    feature_names = dataset['feature_names']

    cfs_ord, \
    cfs_eval, \
    x_cfs_ord, \
    x_cfs_eval = evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                                         toolbox, objective_names, featureScaler, feature_names)


    # recovering counter-factuals in original format
    x_org, \
    cfs_org, \
    x_cfs_org, \
    x_cfs_highlight = recoverOriginals(x_ord, cfs_ord, dataset, feature_names)

    # returning the results
    output = {'cfs_ord': cfs_ord,
                'cfs_org': cfs_org,
                'cfs_eval': cfs_eval,
                'x_cfs_ord': x_cfs_ord,
                'x_cfs_eval': x_cfs_eval,
                'x_cfs_org': x_cfs_org,
                'x_cfs_highlight': x_cfs_highlight,
                }

    return output