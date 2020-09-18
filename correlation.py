import numpy as np

def Correlation(x, cf, theta_cf, corr_models, feature_width, discrete_indices, continuous_indices):

    distance = []
    cf_ = cf.copy()
    delta = np.nonzero(x-cf)[0]
    for m in corr_models:
        feature = m['feature']
        model = m['model']
        inputs = m['inputs']
        score = m['score']
        if feature in delta:
            cf_[feature] = model.predict(theta_cf[inputs].reshape(1, -1))
            if feature in discrete_indices:
                distance.append(score * int(cf[feature] != cf_[feature]))
            elif feature in continuous_indices:
                distance.append(score * (1 / feature_width[feature]) * abs(cf[feature] - cf_[feature]))

    cost = 0 if distance == [] else np.mean(distance)
    return cost

