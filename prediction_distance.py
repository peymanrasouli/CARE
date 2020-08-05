def PredictionDistance(cf, blackbox, output_range, label_cf):
    if label_cf is None:
        resp_cf = blackbox.predict(cf.reshape(1, -1))
        return min(abs(resp_cf - output_range))
    else:
        prob_cf = blackbox.predict_proba(cf.reshape(1,-1))
        if output_range[0] <= prob_cf[0,label_cf] <= output_range[1]:
            return 0
        else:
            return min(abs(prob_cf-output_range)[0])

