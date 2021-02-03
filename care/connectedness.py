import hdbscan

def connectednessObj(cf_theta, connectedness_model):
    test_label, strength = hdbscan.approximate_predict(connectedness_model, cf_theta.reshape(1,-1))
    fitness = int(strength[0] > 0.0)
    return fitness