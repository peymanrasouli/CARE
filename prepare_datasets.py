import pandas as pd
from sklearn.preprocessing import LabelEncoder

def PrepareBreastCancer(dataset_path, file_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+file_name, delimiter=',')

    ## Recognizing inputs
    class_name = 'Class'
    df_X = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    discrete_features = list(df_X.columns)
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_features = []    # none continuous features
    continuous_indices = []     # none continuous features

    ## Encoding features
    df_X_fe = df_X.copy(deep=True)
    feature_encoder = dict()
    for col in discrete_features:
        fe = LabelEncoder()
        df_X_fe[col] = fe.fit_transform(df_X_fe[col])
        feature_encoder[col] = fe

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X = df_X_fe.loc[:, df_X_fe.columns!=class_name].values
    y = df_y_le

    ## Indexing labels
    label_names = list(df_y.unique())
    label_indices = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {i: [min(X[:,i]),max(X[:,i])] for i in range(X.shape[1])}

    ## Returning dataset information
    dataset = {
        'name': file_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_fe': df_X_fe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_names': label_names,
        'label_indices': label_indices,
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'X': X,
        'y': y
    }

    return dataset