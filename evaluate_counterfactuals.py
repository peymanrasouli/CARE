import pandas as pd
from utils import *

def evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                            toolbox, objective_names, featureScaler, feature_names):

    cfs_ord.drop_duplicates(inplace=True)
    cfs_ord.reset_index(drop=True, inplace=True)

    x_ord = pd.DataFrame(data=x_ord.reshape(1, -1), columns=feature_names)
    x_cfs_ord = pd.concat([x_ord, cfs_ord])
    x_cfs_ord.reset_index(drop=True, inplace=True)

    x_cfs_theta = ord2theta(x_cfs_ord, featureScaler)
    x_cfs_ohe = ord2ohe(x_cfs_ord.to_numpy(), dataset)

    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in x_cfs_theta])
    if task == 'classification':
        label = predict_fn(x_cfs_ohe)
        probability = np.max(predict_proba_fn(x_cfs_ohe), axis=1)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation,label,probability], columns=objective_names+['Class','Probability'])
    else:
        response = predict_fn(x_cfs_ohe)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation, response], columns=objective_names+['Response'])

    cfs_ord_ = x_cfs_ord.iloc[1:,:].copy(deep=True)
    cfs_eval = x_cfs_eval.iloc[1:,:].copy(deep=True)

    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_ord_ = cfs_ord_.set_index(index)
    cfs_eval = cfs_eval.set_index(index)

    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_ord) - 1)])
    x_cfs_ord = x_cfs_ord.set_index(index)
    x_cfs_eval = x_cfs_eval.set_index(index)

    return cfs_ord_, cfs_eval, x_cfs_ord, x_cfs_eval
