import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from user_preferences import userPreferences
from care_explainer import CAREExplainer

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'), # use 'nn-c' or 'gb-c'
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'), # use 'nn-c' or 'gb-c'
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),  # use 'nn-c' or 'gb-c'
        # 'iris': ('iris-sklearn', PrepareIris, 'classification'),  # use 'gb-c'
        # 'diabetes': ('diabetes-sklearn', PrepareDiabetes, 'regression') # use 'nn-r' or 'gb-r'
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression') # use 'nn-r' or 'gb-r'
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        # 'gb-c': GradientBoostingClassifier,
        # 'nn-r': MLPRegressor,
        # 'gb-r': GradientBoostingRegressor
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn-c':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # instance to explain
            ind = 0
            x_ord = X_test[ind]
            n_cf = 10

            # set user preferences || they are taken into account when ACTIONABILITY=True!
            user_preferences = userPreferences(dataset, x_ord)

            # explain instance x_ord using CARE
            output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                   SOUNDNESS=True, CAUSALITY=True, ACTIONABILITY=True,
                                   user_preferences=user_preferences, n_population=200, n_generation=20,
                                   cf_class='neighbor', probability_thresh=0.5, cf_quantile='neighbor', n_cf=n_cf)

            # print counterfactuals and their corresponding objective values
            print('\n')
            print(output['x_cfs_highlight'])
            print(output['x_cfs_eval'])

            print('\n')
            print('Done!')

if __name__ == '__main__':
    main()
