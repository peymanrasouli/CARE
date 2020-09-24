import numpy as np
import pandas as pd

def EvaluateCounterfactuals(cfs, solutions, blackbox, toolbox, OBJ_name, task):

    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in solutions])
    if task == 'classification':
        label = blackbox.predict(cfs)
        probability = np.max(blackbox.predict_proba(cfs), axis=1)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation,label,probability], columns=OBJ_name+['Label','Probability'])
    else:
        response = blackbox.predict(cfs)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation, response], columns=OBJ_name+['Response'])

    drop_ind = cfs_eval[cfs_eval['Prediction']>0].index

    cfs.drop(drop_ind, inplace=True)
    cfs.reset_index(drop=True, inplace=True)

    cfs_eval.drop(drop_ind, inplace=True)
    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_eval = cfs_eval.set_index(index)

    return cfs, cfs_eval
