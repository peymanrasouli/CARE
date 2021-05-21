from care.care import CARE
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn, explainer=None,
                  SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False, user_preferences=None,
                  n_population=200, n_generation=20, cf_class='opposite', probability_thresh=0.5,
                  cf_quantile='neighbor', n_cf=5):

    # creating an explainer instance in case it is not pre-created
    if explainer is None:

        # creating an instance of CARE explainer
        explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                         SOUNDNESS=SOUNDNESS, COHERENCY=COHERENCY, ACTIONABILITY=ACTIONABILITY,
                         n_population=n_population, n_generation=n_generation, n_cf=n_cf)

        # fitting the explainer on the training data
        explainer.fit(X_train, Y_train)

    # generating counterfactuals
    explanations = explainer.explain(x_ord, cf_class=cf_class, cf_quantile=cf_quantile,
                                     probability_thresh=probability_thresh,
                                     user_preferences=user_preferences)

    # extracting results
    cfs_ord = explanations['cfs_ord']
    toolbox = explanations['toolbox']
    objective_names = explanations['objective_names']
    featureScaler = explanations['featureScaler']
    feature_names = dataset['feature_names']

    # evaluating counterfactuals
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
              'toolbox': toolbox,
              'featureScaler': featureScaler,
              'objective_names': objective_names,
              }

    return output