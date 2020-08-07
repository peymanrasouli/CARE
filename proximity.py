def Proximity(theta_cf, lof_model):
    outlier = lof_model.predict(theta_cf.reshape(1,-1))[0]
    return outlier
