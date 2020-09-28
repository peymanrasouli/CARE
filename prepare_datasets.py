import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler

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
    continuous_features = []    # none continuous features

    df_X = pd.concat([df_X[discrete_features], df_X[continuous_features]], axis=1)
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    # Scaling continuous features
    num_feature_scaler = None

    ## Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    ## One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = ord_encoded_data
    df_X_ohe = ohe_encoded_data

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_discrete_ord = [0, ord_encoded_data.shape[1]]
    len_continuous_ord = []

    len_discrete_ohe = [0, ohe_encoded_data.shape[1]]
    len_continuous_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
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
    continuous_features = ['LIMIT_BAL', 'AGE',  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                           'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    df_X = pd.concat([df_X[discrete_features], df_X[continuous_features]], axis=1)
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    # Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    ## One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([ord_encoded_data, scaled_data], axis=1)
    df_X_ohe = pd.concat([ohe_encoded_data, scaled_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_discrete_ord = [0, ord_encoded_data.shape[1]]
    len_continuous_ord = [ord_encoded_data.shape[1] + 1, df_X_ord.shape[1]]

    len_discrete_ohe = [0, ohe_encoded_data.shape[1]]
    len_continuous_ohe = [ohe_encoded_data.shape[1] + 1, df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
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
    continuous_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

    df_X = pd.concat([df_X[discrete_features],df_X[continuous_features]], axis=1)
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    # Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    ## One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([ord_encoded_data, scaled_data], axis=1)
    df_X_ohe = pd.concat([ohe_encoded_data, scaled_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:,i]),max(X_ord[:,i])] for i in range(X_ord.shape[1])}

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_discrete_ord = [0, ord_encoded_data.shape[1]]
    len_continuous_ord = [ord_encoded_data.shape[1] + 1, df_X_ord.shape[1]]

    len_discrete_ohe = [0, ohe_encoded_data.shape[1]]
    len_continuous_ohe = [ohe_encoded_data.shape[1] + 1 , df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
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
    df_y = df.loc[:, target_name]

    discrete_features = ['CHAS']
    continuous_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT']

    df_X = pd.concat([df_X[discrete_features], df_X[continuous_features]], axis=1)
    discrete_indices = [df_X.columns.get_loc(f) for f in discrete_features]
    continuous_indices = [df_X.columns.get_loc(f) for f in continuous_features]

    # Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    ## One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(df_X.iloc[:, discrete_indices].to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([ord_encoded_data, scaled_data], axis=1)
    df_X_ohe = pd.concat([ohe_encoded_data, scaled_data], axis=1)

    ## Extracting raw data and labels
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y =  df_y.to_numpy()

    ## Extracting target range
    target_range = [min(y),max(y)]

    ## Indexing features
    feature_names = list(df_X.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_discrete_ord = [0, ord_encoded_data.shape[1]]
    len_continuous_ord = [ord_encoded_data.shape[1] + 1, df_X_ord.shape[1]]

    len_discrete_ohe = [0, ohe_encoded_data.shape[1]]
    len_continuous_ohe = [ohe_encoded_data.shape[1] + 1, df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_X': df_X,
        'df_y': df_y,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'target_name': target_name,
        'target_range': target_range,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset

