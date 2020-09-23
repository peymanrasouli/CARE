import numpy as np

def Correlation(x_bb, cf_bb, cf_theta, corr_models, feature_width, discrete_indices, continuous_indices):

    distance = []
    cf_bb_ = cf_bb.copy()
    delta = np.nonzero(x_bb-cf_bb)[0]
    for m in corr_models:
        feature = m['feature']
        model = m['model']
        inputs = m['inputs']
        score = m['score']
        if feature in delta:
            cf_bb_[feature] = model.predict(cf_theta[inputs].reshape(1, -1))
            if feature in discrete_indices:
                distance.append(score * int(cf_bb[feature] != cf_bb_[feature]))
            elif feature in continuous_indices:
                distance.append(score * (1 / feature_width[feature]) * abs(cf_bb[feature] - cf_bb_[feature]))

    cost = 0 if distance == [] else np.mean(distance)
    return cost

