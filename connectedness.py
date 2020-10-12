import hdbscan

def connectedness(cf_theta, connectedness_model):
    test_label, strength = hdbscan.approximate_predict(connectedness_model, cf_theta.reshape(1,-1))
    return strength[0]