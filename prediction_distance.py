import numpy as np

def PredictionDistance(cf, blackbox, probability_thresh, cf_label, cf_range):
    if cf_label is None:
        cf_response = blackbox.predict(cf.reshape(1, -1))
        cost = 0 if np.logical_and(cf_response>=cf_range[0], cf_response<=cf_range[1]) else\
            min(abs(cf_response - cf_range))
        return cost
    else:
        cf_probability = blackbox.predict_proba(cf.reshape(1, -1))[0,cf_label]
        cost = np.max([0,probability_thresh-cf_probability])
        return cost