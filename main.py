import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, KerasNeuralNetwork
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from user_preferences import userPreferences
from mocf_explainer import MOCFExplainer

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'dnn': KerasNeuralNetwork,
        # 'gt': GradientBoostingClassifier,
        # 'mlp-r': MLPRegressor
        # 'dt-r': DecisionTreeRegressor,
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

            # Creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name is 'dnn':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # classification task
            if task is 'classification':

                # instance to explain
                ind = 0
                x_ord = X_test[ind]

                # set user preferences
                user_preferences = userPreferences(dataset, x_ord)

                # explain instance x_ord using MOCF
                output = MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                        soundCF=False, feasibleAR=False, user_preferences=user_preferences,
                                        cf_class='opposite', probability_thresh=0.5)

                # print n best counter-factuals and their corresponding objective values
                n = 5
                print('\n')
                print(output['x_cfs_highlight'].head(n= n + 1))
                print(output['x_cfs_eval'].head(n= n + 1))

                print('\n')
                print('Done!')

            # regression task
            elif task is 'regression':

                # instance to explain
                ind = 0
                x_ord = X_test[ind]

                # set user preferences
                user_preferences = userPreferences(dataset, x_ord)

                # explain instance x_ord using MOCF
                output = MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                        soundCF=False, feasibleAR=False, user_preferences=user_preferences,
                                        cf_quantile='neighbor')

                # print n best counter-factuals and their corresponding objective values
                n = 5
                print('\n')
                print(output['x_cfs_highlight'].head(n= n + 1))
                print(output['x_cfs_eval'].head(n= n + 1))

                print('\n')
                print('Done!')

if __name__ == '__main__':
    main()
