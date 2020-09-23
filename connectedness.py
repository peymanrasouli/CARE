import hdbscan

def Connectedness(cf_theta, hdbscan_model):
    test_label, strength = hdbscan.approximate_predict(hdbscan_model, cf_theta.reshape(1,-1))
    return strength[0]