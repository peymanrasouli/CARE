def proximity(cf_theta, proximity_model):
    status = proximity_model.predict(cf_theta.reshape(1, -1))[0]
    fitness = max(0, status)
    return fitness
