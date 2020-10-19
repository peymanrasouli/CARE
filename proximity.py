def proximity(cf_theta, proximity_model):
    outlier = proximity_model.predict(cf_theta.reshape(1,-1))[0]
    return max(0, outlier)
