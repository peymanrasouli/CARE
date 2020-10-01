import numpy as np

def ord2ohe(X_ord, dataset):
    ohe_feature_encoder = dataset['ohe_feature_encoder']
    len_continuous_ord = dataset['len_continuous_ord']
    len_discrete_ord = dataset['len_discrete_ord']

    if X_ord.shape.__len__() == 1:
        X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
        X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
        X_discrete = ohe_feature_encoder.transform(X_discrete.reshape(1,-1)).ravel()
        X_ohe = np.r_[X_continuous, X_discrete]
        return X_ohe
    else:
        X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
        X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
        X_discrete = ohe_feature_encoder.transform(X_discrete)
        X_ohe = np.c_[X_continuous,X_discrete]
        return X_ohe

def ord2org(X_ord, dataset):
    num_feature_scaler = dataset['num_feature_scaler']
    ord_feature_encoder = dataset['ord_feature_encoder']
    len_continuous_ord = dataset['len_continuous_ord']
    len_discrete_ord = dataset['len_discrete_ord']

    if X_ord.shape.__len__() == 1:
        X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
        X_continuous = num_feature_scaler.inverse_transform(X_continuous.reshape(1, -1)).ravel()
        X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
        X_discrete = ord_feature_encoder.inverse_transform(X_discrete.reshape(1,-1)).ravel()
        X_org = np.r_[X_continuous, X_discrete]
        return X_org
    else:
        X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
        X_continuous = num_feature_scaler.inverse_transform(X_continuous)
        X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
        X_discrete = ord_feature_encoder.inverse_transform(X_discrete)
        X_org = np.c_[X_continuous,X_discrete]
        return X_org

def ord2theta(X_ord, ea_scaler):
    if X_ord.shape.__len__() == 1:
        X_theta = ea_scaler.transform(X_ord.reshape(1,-1)).ravel()
        return X_theta
    else:
        X_theta = ea_scaler.transform(X_ord)
        return X_theta

def theta2ord(X_theta, ea_scaler, discrete_indices):
    if X_theta.shape.__len__() == 1:
        X_ord = ea_scaler.inverse_transform(X_theta.reshape(1,-1)).ravel()
        X_ord[discrete_indices] = np.rint(X_ord[discrete_indices])
        return X_ord
    else:
        X_ord = ea_scaler.inverse_transform(X_theta)
        X_ord[:, discrete_indices] = np.rint(X_ord[:, discrete_indices])
        return X_ord