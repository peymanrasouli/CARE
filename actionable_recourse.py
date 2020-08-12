import numpy as np
def ActionableRecourse(x, cf, actions_op, actions_wt):
    status = [int(not(a(cf[f], x[f]))) if type(a) == np.ufunc else a for f,a in enumerate(actions_op)]
    cost = sum(np.asarray(status) * np.asarray(actions_wt))
    return cost