import numpy as np

def PredictionDistance(cf_ohe, predict_class_fn, predict_proba_fn, probability_thresh, cf_class, cf_range):
    if cf_class is None:
        cf_response = predict_class_fn(cf_ohe.reshape(1, -1))
        cost = 0 if np.logical_and(cf_response >= cf_range[0], cf_response <= cf_range[1]) else\
            min(abs(cf_response - cf_range))
        return cost
    else:
        cf_probability = predict_proba_fn(cf_ohe.reshape(1, -1))[0, cf_class]
        cost = np.max([0,probability_thresh-cf_probability])
        return cost