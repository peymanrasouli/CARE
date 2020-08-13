import numpy as np

def ActionableRecourse(x, cf, actions_o, actions_w):
    status = [int(not(a(cf[f], x[f]))) if type(a) == np.ufunc else a for f, a in enumerate(actions_o)]
    cost = sum(np.asarray(status) * np.asarray(actions_w))
    return cost