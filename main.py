from prepare_datasets import *
from mocf_explainer import MOCFExplainer
from cf_explainer import CFExplainer
from cf_prototype_explainer import CFPrototypeExplainer
from create_model import CreateModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main():
    ## Defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    ## Defining the list of data sets
    datsets_list = {
        # 'breast-cancer': ('breast-cancer.csv', PrepareBreastCancer, 'classification'),
        'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression')
    }

    ## Defining the list of black-boxes
    blackbox_list = {
        # 'lg': LogisticRegression,
        # 'gt': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        'nn': MLPClassifier,
        # 'dtr': DecisionTreeRegressor,
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        ## Reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        ## Splitting the data set into train and test sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            ## Creating black-box model
            blackbox = CreateModel(X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)

            ## Explaining the instance using counter-factuals
            # Classification
            if task is 'classification':
                ind = 0
                x = X_test[ind]
                x_label = blackbox.predict(x.reshape(1, -1))
                cf_label = int(1 - x_label)      # Counter-factual label
                probability_thresh = 0.6         # Desired probability threshold

                ## MOCF Explainer
                MOCF_output = MOCFExplainer(x, blackbox, dataset, task, X_train, Y_train,
                                            probability_thresh=probability_thresh, cf_label=cf_label)

                ## CF Explainer
                CF_output = CFExplainer(x, blackbox, dataset, task, probability_thresh, MOCF_output)

                ## CFProto Explainer
                CFProto_output = CFPrototypeExplainer(x, blackbox, X_train, dataset, task, MOCF_output)

                print('Done!')

            # Regression
            elif task is 'regression':
                def SelectResponseRange(x, blackbox, dataset):
                    q = np.quantile(dataset['y'], q=np.linspace(0,1,11))
                    ranges = [[q[i], q[i+1]] for i in range(len(q)-1)]
                    response_x = blackbox.predict(x.reshape(1, -1))
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
                x_range, cf_range = SelectResponseRange(x, blackbox, dataset)    # Desired response range

                ## MOCF Explainer
                MOCF_output = MOCFExplainer(x, blackbox, dataset, task, X_train, Y_train,
                                            x_range = x_range, cf_range=cf_range)

                ## CF Explainer

                ## CFProto Explainer

if __name__ == '__main__':
    main()
