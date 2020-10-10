import numpy as np
import pandas as pd
from mappings import ord2theta, ord2ohe

def evaluateCounterfactuals(cfs, dataset, predict_fn, predict_proba_fn, task, toolbox,
                            objective_names, objective_weights, eaScaler):

    cfs.drop_duplicates(inplace=True)
    cfs.reset_index(drop=True, inplace=True)
    cfs_theta = ord2theta(cfs, eaScaler)
    cfs_ohe = ord2ohe(cfs.to_numpy(), dataset)

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
    cfs = cfs.reindex(cfs_eval.index)

    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_eval = cfs_eval.set_index(index)

    return cfs, cfs_eval
