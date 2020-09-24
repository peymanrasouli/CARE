import numpy as np
import pandas as pd

def EvaluateCounterfactuals(cfs, solutions, toolbox, OBJ_name):

    evaluation = [np.asarray(toolbox.evaluate(ind)) for ind in solutions]
    cfs_eval = pd.DataFrame(data=evaluation, columns=OBJ_name)

    drop_ind = cfs_eval[cfs_eval['Prediction']>0].index

    cfs.drop(drop_ind, inplace=True)
    cfs.reset_index(drop=True, inplace=True)

    cfs_eval.drop(drop_ind, inplace=True)
    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_eval = cfs_eval.set_index(index)

    return cfs, cfs_eval
