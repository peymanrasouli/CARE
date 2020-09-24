import pandas as pd
import numpy as np

def BB2Theta(bb, ea_scaler):
    try:
        theta = ea_scaler.transform(bb.reshape(1, -1)).ravel()

    except Exception:
        theta = ea_scaler.transform(bb)

    return theta

def Theta2BB(theta, ea_scaler):
    try:
        theta = ea_scaler.inverse_transform(theta.reshape(1, -1)).ravel()

    except Exception:
        theta = ea_scaler.inverse_transform(theta)

    return theta

def BB2Original(bb, feature_encoder, feature_scaler, discrete_indices, continuous_indices):
    try:
        original = pd.DataFrame(data=np.zeros([1,len(bb)]))
        if len(discrete_indices):
            decoded_data = feature_encoder.inverse_transform(bb[discrete_indices].reshape(1,-1)).ravel()
            original.iloc[0,discrete_indices] = decoded_data

        if len(continuous_indices):
            descaled_data = feature_scaler.inverse_transform(bb[continuous_indices].reshape(1,-1)).ravel()
            original.iloc[0,continuous_indices] = descaled_data

    except Exception:
        original = pd.DataFrame(data=np.zeros(bb.shape))
        if len(discrete_indices):
            decoded_data = feature_encoder.inverse_transform(bb.iloc[:,discrete_indices])
            original.iloc[:,discrete_indices] = decoded_data

        if len(continuous_indices):
            descaled_data = feature_scaler.inverse_transform(bb.iloc[:,continuous_indices])
            original.iloc[:,continuous_indices] = descaled_data

    return original