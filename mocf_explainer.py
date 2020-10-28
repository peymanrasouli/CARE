from mocf import MOCF
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                  soundCF=False, feasibleAR=False, user_preferences=None,
                  cf_class='opposite', probability_thresh=0.5, cf_quantile='neighbor', n_cf=5):

    # creating an instance of MOCF explainer
    mocf_explainer = MOCF(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                     soundCF=soundCF, feasibleAR=feasibleAR, n_cf=n_cf)

    # fitting the explainer on the training data
    mocf_explainer.fit(X_train, Y_train)

    # generating counter-factuals
    explanations = mocf_explainer.explain(x_ord, cf_class=cf_class, cf_quantile=cf_quantile,
                                    probability_thresh=probability_thresh, user_preferences=user_preferences)

    # extracting results
    cfs_ord = explanations['cfs_ord']
    toolbox = explanations['toolbox']
    objective_names = explanations['objective_names']
    featureScaler = explanations['featureScaler']
    feature_names = dataset['feature_names']

    # evaluating counter-factuals
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

    # best counter-factual
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
              'toolbox': toolbox,
              'featureScaler': featureScaler,
              'objective_names': objective_names,
              }

    return output