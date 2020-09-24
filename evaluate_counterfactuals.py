import numpy as np
import pandas as pd

def EvaluateCounterfactuals(cfs, toolbox, OBJ_name, ea_scaler, discrete_indices):

    data_theta = cfs.copy(deep=True)
    data_theta = data_theta.to_numpy()
    data_theta[:,discrete_indices] = ea_scaler.transform(data_theta[:,discrete_indices])

    evaluation = [np.asarray(toolbox.evaluate(ind)) for ind in data_theta]
    cfs_eval = pd.DataFrame(data=evaluation, columns=OBJ_name)

    drop_ind = cfs_eval[cfs_eval['Prediction']>0].index

    cfs.drop(drop_ind, inplace=True)
    cfs.reset_index(drop=True, inplace=True)

    cfs_eval.drop(drop_ind, inplace=True)
    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_eval = cfs_eval.set_index(index)

    return cfs, cfs_eval
