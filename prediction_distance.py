def PredictionDistance(cf, blackbox, probability_range, response_range, cf_label):
    if cf_label is None:
        cf_response = blackbox.predict(cf.reshape(1, -1))
        return min(abs(cf_response - response_range))
    else:
        cf_probability = blackbox.predict_proba(cf.reshape(1, -1))[0,cf_label]
        if probability_range[0] <= cf_probability <= probability_range[1]:
            return 0
        else:
            return min(abs(cf_probability - probability_range))