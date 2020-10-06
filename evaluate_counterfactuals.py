import numpy as np
import pandas as pd
from mappings import ord2theta, ord2ohe

def EvaluateCounterfactuals(cfs, dataset, predict_class_fn, predict_proba_fn, task, MOCF_output):

    toolbox = MOCF_output['toolbox']
    OBJ_name = MOCF_output['OBJ_name']
    OBJ_order = MOCF_output['OBJ_order']
    OBJ_W = MOCF_output['OBJ_W']
    ea_scaler = MOCF_output['ea_scaler']

    cfs.drop_duplicates(inplace=True)
    cfs.reset_index(drop=True, inplace=True)
    cfs_theta = ord2theta(cfs, ea_scaler)
    cfs_ohe = ord2ohe(cfs.to_numpy(), dataset)

    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in cfs_theta])
    if task == 'classification':
        label = predict_class_fn(cfs_ohe)
        probability = np.max(predict_proba_fn(cfs_ohe), axis=1)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation,label,probability], columns=OBJ_name+['Class','Probability'])
    else:
        response = predict_class_fn(cfs_ohe)
        cfs_eval = pd.DataFrame(data=np.c_[evaluation, response], columns=OBJ_name+['Response'])


    OBJ_name_orderd = [OBJ_name[ord-1] for ord in OBJ_order]
    OBJ_W_ordered = [False if OBJ_W[ord-1] == 1.0 else True for ord in OBJ_order]

    cfs_eval = cfs_eval.sort_values(by=OBJ_name_orderd,ascending=OBJ_W_ordered)
    cfs = cfs.reindex(cfs_eval.index)

    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_eval = cfs_eval.set_index(index)

    return cfs, cfs_eval
