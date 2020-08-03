def PredictionDistance(cf,l_cf, blackbox, probability_range):
    prob_cf = blackbox.predict_proba(cf.reshape(1,-1))[0,l_cf]
    if probability_range[0] <= prob_cf <= probability_range[1]:
        return 0
    else:
        return abs(prob_cf-probability_range[0])

