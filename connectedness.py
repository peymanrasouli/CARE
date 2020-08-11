import hdbscan

def Connectedness(theta_cf, hdbscan_model):
    test_label, strength = hdbscan.approximate_predict(hdbscan_model, theta_cf.reshape(1,-1))
    return strength[0]