def Proximity(cf_theta, lof_model):
    outlier = lof_model.predict(cf_theta.reshape(1,-1))[0]
    return outlier
