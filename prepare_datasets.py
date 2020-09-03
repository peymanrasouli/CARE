import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

## Preparing Breast Cancer dataset
def PrepareBreastCancer(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=',', na_values = '?')

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'Class'
    df_X = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    discrete_features = ['age', 'menopause', 'tumor-size', 'inv-node', 'node-caps', 'deg-malig', 'breast',
                         'breast-quad', 'irradiat']
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_features = []    # none continuous features
    continuous_indices = []     # none continuous features

    ## Encoding features
    df_X_fe = df_X.copy(deep=True)
    feature_encoder = dict()
    for col in discrete_features:
        fe = OrdinalEncoder()
        encoded_data = fe.fit_transform(df_X_fe[col].to_numpy().reshape(-1, 1))
        df_X_fe[col] = pd.DataFrame(encoded_data)
        feature_encoder[col] = fe

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X = df_X_fe.values
    y = df_y_le

    ## Indexing labels
    label_names = list(df_y.unique())
    label_indices = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X[:,i]),max(X[:,i])] for i in range(X.shape[1])}

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_fe': df_X_fe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'label_names': label_names,
        'label_indices': label_indices,
        'feature_encoder': feature_encoder,
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

## Preparing Default of Credit Card Clients dataset
def PrepareCreditCardDefault(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=',')

    ## Recognizing inputs
    class_name = 'default payment next month'
    df_X = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    discrete_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_features = ['LIMIT_BAL', 'AGE',  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                           'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    ## Encoding features
    df_X_fe = df_X.copy(deep=True)
    feature_encoder = dict()
    for col in discrete_features:
        fe = OrdinalEncoder()
        encoded_data = fe.fit_transform(df_X_fe[col].to_numpy().reshape(-1, 1))
        df_X_fe[col] = pd.DataFrame(encoded_data)
        feature_encoder[col] = fe

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X = df_X_fe.values
    y = df_y_le

    ## Indexing labels
    label_names = list(df_y.unique())
    label_indices = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X[:,i]),max(X[:,i])] for i in range(X.shape[1])}

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_fe': df_X_fe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'label_names': label_names,
        'label_indices': label_indices,
        'feature_encoder': feature_encoder,
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

## Preparing Adult dataset
def PrepareAdult(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values=' ?')

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df_X = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    discrete_features = ['work-class', 'education','education-num', 'marital-status', 'occupation', 'relationship', 'race',
                         'sex', 'native-country']
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    ## Encoding features
    df_X_fe = df_X.copy(deep=True)
    feature_encoder = dict()
    for col in discrete_features:
        fe = OrdinalEncoder()
        encoded_data = fe.fit_transform(df_X_fe[col].to_numpy().reshape(-1, 1))
        df_X_fe[col] = pd.DataFrame(encoded_data)
        feature_encoder[col] = fe

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X = df_X_fe.values
    y = df_y_le

    ## Indexing labels
    label_names = list(df_y.unique())
    label_indices = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X[:,i]),max(X[:,i])] for i in range(X.shape[1])}

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_fe': df_X_fe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'label_names': label_names,
        'label_indices': label_indices,
        'feature_encoder': feature_encoder,
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

## Preparing Boston House Prices dataset
def PrepareBostonHousePrices(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values=' ?')

    ## Recognizing inputs
    target_name = 'MEDV'
    df_X = df.loc[:, df.columns != target_name]
    df_X = (df_X.round(5)*100000).astype(int)
    df_y = df.loc[:, target_name]

    discrete_features = ['CHAS']
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT']
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    ## Encoding features
    df_X_fe = df_X.copy(deep=True)
    feature_encoder = dict()
    for col in discrete_features:
        fe = OrdinalEncoder()
        encoded_data = fe.fit_transform(df_X_fe[col].to_numpy().reshape(-1, 1))
        df_X_fe[col] = pd.DataFrame(encoded_data)
        feature_encoder[col] = fe

    ## Extracting raw data and target
    X = df_X_fe.values
    y = df_y.to_numpy()

    ## Extracting target range
    target_range = [min(y),max(y)]

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X[:,i]),max(X[:,i])] for i in range(X.shape[1])}

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_fe': df_X_fe,
        'target_name': target_name,
        'target_range': target_range,
        'feature_encoder': feature_encoder,
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
