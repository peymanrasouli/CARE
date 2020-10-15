import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, KerasNeuralNetwork
from user_preferences import userPreferences
from mocf_explainer import MOCFExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'dnn': KerasNeuralNetwork
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
            predict_fn = lambda x: blackbox.predict_classes(x).ravel()
            predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()

            # instance to explain
            ind = 0
            x_ord = X_test[ind]

            # set user preferences
            user_preferences = userPreferences(dataset, x_ord)

            # explain instance x_ord using MOCF
            MOCF_output = MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                        soundCF=False, feasibleAR=False, hof_final=False,
                                        user_preferences=user_preferences, cf_class='opposite', probability_thresh=0.5)

            # explain instance x_ord using CFPrototype
            CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset, task,
                                                      MOCF_output, target_class=None)

            # explain instance x_ord using DiCE
            DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train, dataset,
                                        task, MOCF_output, feasibleAR=False, user_preferences=user_preferences,
                                        n_cf=10, desired_class="opposite", probability_thresh=0.5)

            # print n best counter-factuals and their corresponding objective values
            n = 10

            print('\n')
            print('MOCF counter-factuals')
            print(MOCF_output['x_cfs_highlight'].head(n= n + 1))
            print(MOCF_output['x_cfs_eval'].head(n= n + 1))

            print('\n')
            print('CFPrototype counter-factuals')
            print(CFPrototype_output['x_cfs_highlight'].head(n=n + 1))
            print(CFPrototype_output['x_cfs_eval'].head(n=n + 1))

            print('\n')
            print('DiCE counter-factuals')
            print(DiCE_output['x_cfs_highlight'].head(n=n + 1))
            print(DiCE_output['x_cfs_eval'].head(n=n + 1))

            print('\n')
            print('Done!')


if __name__ == '__main__':
    main()
