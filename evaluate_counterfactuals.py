import pandas as pd
from utils import *

def evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                            toolbox, objective_names, featureScaler, feature_names):

    # merging x with counterfactuals
    x_ord = pd.DataFrame(data=x_ord.reshape(1, -1), columns=feature_names)
    x_cfs_ord = pd.concat([x_ord, cfs_ord])
    x_cfs_ord.reset_index(drop=True, inplace=True)

    # mapping data to required formats for evaluation and prediction
    x_cfs_theta = ord2theta(x_cfs_ord, featureScaler)
    x_cfs_ohe = ord2ohe(x_cfs_ord.to_numpy(), dataset)

    # evaluating counterfactuals
    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in x_cfs_theta])

    # checking the validity of counterfactulas (i.e., correct prediction)
    validity = (evaluation[:, 0] == 0.0).astype(int)

    # measuring the diversity of counterfacutuals using Jaccard coefficient
    n_cf = cfs_ord.shape[0]
    if n_cf == 1:
        jaccard = 0.0
    else:
        changed_feature = []
        for i in range(n_cf):
            changed_feature.append([dataset['feature_names'][ii] for ii in
                                    np.where(x_cfs_ord.iloc[0] != x_cfs_ord.iloc[i + 1])[0]])
        changed_feature = list(filter(None, changed_feature))
        jaccard = []
        for i in range(0, len(changed_feature)):
            for ii in range(i, len(changed_feature)):
                similarity = len(set(changed_feature[i]) & set(changed_feature[ii])) / \
                             len(set(changed_feature[i]) | set(changed_feature[ii]))
                jaccard.append(similarity)
        jaccard = np.mean(jaccard)
    diversity = [(1.0 - jaccard) for _ in range(n_cf + 1)]

    # creating an evaluation data frame
    if task == 'classification':
        label = predict_fn(x_cfs_ohe)
        probability = np.max(predict_proba_fn(x_cfs_ohe), axis=1)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation, validity, diversity, label, probability],
                                  columns=objective_names+['Validity', 'Diversity', 'Class', 'Probability'])
    else:
        response = predict_fn(x_cfs_ohe)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation, validity, diversity, response],
                                  columns=objective_names+['Validity', 'Diversity', 'Response'])

    # creating data frames for counterfactuals
    cfs_ord_ = x_cfs_ord.iloc[1:,:].copy(deep=True)
    cfs_eval = x_cfs_eval.iloc[1:,:].copy(deep=True)

    # indexing counterfactual data frames
    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_ord_ = cfs_ord_.set_index(index)
    cfs_eval = cfs_eval.set_index(index)

    # indexing x+counterfactual data frames
    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_ord) - 1)])
    x_cfs_ord = x_cfs_ord.set_index(index)
    x_cfs_eval = x_cfs_eval.set_index(index)

    return cfs_ord_, cfs_eval, x_cfs_ord, x_cfs_eval