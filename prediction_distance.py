import numpy as np

def PredictionDistance(cf, blackbox, probability_thresh, response_range, cf_label):
    if cf_label is None:
        cf_response = blackbox.predict(cf.reshape(1, -1))
        return min(abs(cf_response - response_range))
    else:
        cf_probability = blackbox.predict_proba(cf.reshape(1, -1))[0,cf_label]
        return np.max([0,probability_thresh-cf_probability])