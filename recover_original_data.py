import pandas as pd

def RecoverOriginalData(x_bb, df, dataset):
    feature_names =  dataset['feature_names']
    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    feature_encoder = dataset['feature_encoder']
    feature_scaler = dataset['feature_scaler']

    df_original = pd.DataFrame(data=x_bb.reshape(1,-1), columns=feature_names)
    df_original = df_original.append(df)

    index = pd.Series(['x'] + ['cf_'+str(i) for i in range(len(df))])
    df_original = df_original.set_index(index)

    if len(discrete_indices):
        decoded_data = feature_encoder.inverse_transform(df_original.iloc[:, discrete_indices])
        df_original.iloc[:, discrete_indices] = decoded_data

    if len(continuous_indices):
        descaled_data = feature_scaler.inverse_transform(df_original.iloc[:, continuous_indices])
        df_original.iloc[:, continuous_indices] = descaled_data

    return df_original
