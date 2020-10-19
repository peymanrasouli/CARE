from mocf import MOCF
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                  soundCF=False, feasibleAR=False, hof_final=False, user_preferences=None,
                  cf_class='opposite', probability_thresh=0.5, cf_quantile='neighbor'):

    # creating an instance of MOCF explainer
    explainer = MOCF(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                     soundCF=soundCF, feasibleAR=feasibleAR, hof_final=hof_final)

    # fitting the explainer on the training data
    explainer.fit(X_train, Y_train)

    # generating counter-factuals
    explanation = explainer.explain(x_ord, cf_class=cf_class, cf_quantile=cf_quantile,
                                    probability_thresh=probability_thresh, user_preferences=user_preferences)

    # extracting results
    cfs_ord = explanation['cfs_ord']
    toolbox = explanation['toolbox']
    objective_names = explanation['objective_names']
    featureScaler = explanation['featureScaler']
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

    # returning the results
    output = {'cfs_ord': cfs_ord,
                'cfs_org': cfs_org,
                'cfs_eval': cfs_eval,
                'x_cfs_ord': x_cfs_ord,
                'x_cfs_eval': x_cfs_eval,
                'x_cfs_org': x_cfs_org,
                'x_cfs_highlight': x_cfs_highlight,
                'toolbox': toolbox,
                'featureScaler': featureScaler,
                'objective_names': objective_names,
                }

    return output