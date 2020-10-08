import warnings
warnings.filterwarnings("ignore")
import numpy as np
from prepare_datasets import *
from mappings import ord2ohe
from mocf_explainer import MOCFExplainer
from dice_explainer import DiCEExplainer
from cf_prototype_explainer import CFPrototypeExplainer
from create_model import CreateModel, KerasNeuralNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def main():
    ## Defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    ## Defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression')
    }

    ## Defining the list of black-boxes
    blackbox_list = {
        'dnn': KerasNeuralNetwork,
        # 'lg': LogisticRegression,
        # 'gt': GradientBoostingClassifier,
        # 'mlp-r': MLPRegressor
        # 'dt-r': DecisionTreeRegressor,
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        ## Reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        ## Splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            ## Creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name is 'dnn':
                predict_class_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_class_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            ## Explaining the instance using counter-factuals
            # Classification
            if task is 'classification':
                ind = 0
                x = X_test[ind]
                x_ohe = ord2ohe(x, dataset)
                x_class = predict_class_fn(x_ohe.reshape(1,-1))
                cf_class = int(1 - x_class)      # Counter-factual class
                probability_thresh = 0.6         # Counter-factual probability threshold

                ## MOCF Explainer
                MOCF_output = MOCFExplainer(x, blackbox, predict_class_fn, predict_proba_fn, dataset, task, X_train,
                                            Y_train, cf_class=cf_class, probability_thresh=probability_thresh)

                ## CFProto Explainer
                CFProto_output = CFPrototypeExplainer(x, predict_class_fn, predict_proba_fn, X_train, dataset, task,
                                                      MOCF_output=MOCF_output)

                ## DiCE Explainer
                DiCE_output = DiCEExplainer(x, blackbox, predict_class_fn, predict_proba_fn, X_train, Y_train, dataset,
                                            task, MOCF_output=MOCF_output, n_cf=10, probability_thresh=probability_thresh)

                print('Done!')

            # Regression
            elif task is 'regression':
                def SelectResponseRange(x, predict_class_fn, dataset):
                    q = np.quantile(dataset['y'], q=np.linspace(0,1,11))
                    ranges = [[q[i], q[i+1]] for i in range(len(q)-1)]
                    response_x = predict_class_fn(x.reshape(1, -1))
                    x_range = -1
                    for i in range(len(ranges)):
                        if ranges[i][0] <= response_x <= ranges[i][1]:
                            x_range = i
                            break
                    cf_range = 1 if x_range == 0 else x_range + np.random.choice([-1,1],1)
                    x_range = ranges[int(x_range)]
                    cf_range = ranges[int(cf_range)]
                    return x_range, cf_range

                ind = 0
                x = X_test[ind]
                x_ohe = ord2ohe(x, dataset)
                x_range, cf_range = SelectResponseRange(x_ohe, predict_class_fn, dataset)    # Desired response range

                ## MOCF Explainer
                MOCF_output = MOCFExplainer(x, blackbox, predict_class_fn, predict_proba_fn, dataset, task, X_train,
                                            Y_train, x_range = x_range, cf_range=cf_range)

                print('Done!')
if __name__ == '__main__':
    main()
