import pandas as pd
from utils import *

def evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task, toolbox,
                            objective_names, objective_weights, featureScaler, feature_names):

    cfs_ord.drop_duplicates(inplace=True)
    cfs_ord.reset_index(drop=True, inplace=True)

    x_ord = pd.DataFrame(data=x_ord.reshape(1, -1), columns=feature_names)
    cfs_ord = pd.concat([x_ord, cfs_ord])
    cfs_ord.reset_index(drop=True, inplace=True)

    cfs_theta = ord2theta(cfs_ord, featureScaler)
    cfs_ohe = ord2ohe(cfs_ord.to_numpy(), dataset)

    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in cfs_theta])
    if task == 'classification':
        label = predict_fn(cfs_ohe)
        probability = np.max(predict_proba_fn(cfs_ohe), axis=1)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation,label,probability], columns=objective_names+['Class','Probability'])
    else:
        response = predict_fn(cfs_ohe)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation, response], columns=objective_names+['Response'])

    objective_weights = [False if objective_weights[i] == 1.0 else True for i in range(len(objective_weights))]
    cfs_eval = cfs_eval.sort_values(by=objective_names, ascending=objective_weights)
    cfs_ord = cfs_ord.reindex(cfs_eval.index)

    x_cfs_ord = pd.concat([cfs_ord.loc[[0]], cfs_ord])
    x_cfs_eval = pd.concat([cfs_eval.loc[[0]], cfs_eval])

    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_ord = cfs_ord.set_index(index)
    cfs_eval = cfs_eval.set_index(index)

    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_ord) - 1)])
    x_cfs_ord = x_cfs_ord.set_index(index)
    x_cfs_eval = x_cfs_eval.set_index(index)

    return cfs_ord, cfs_eval, x_cfs_ord, x_cfs_eval
