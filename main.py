import numpy as np
from mocf import MOCF
from prepare_datasets import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def main():
    ## Defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    ## Defining the list of data sets
    datsets_list = {
        # 'breast_cancer': ('breast-cancer.csv', PrepareBreastCancer),
        'credit_card_default': ('credit-card-default.csv', PrepareCreditCardDefault),
        # 'adult': ('adult.csv', PrepareAdult),
        # 'boston_house_prices': ('boston-house-prices.csv', PrepareBostonHousePrices)
    }

    ## Defining the list of black-boxes
    blackbox_list = {
        # 'lg': LogisticRegression,
        'gt': GradientBoostingClassifier,
        # 'nn': MLPClassifier,
        # 'rfr': RandomForestRegressor
        # 'nnr': MLPRegressor
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        ## Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        ## Splitting the data set into train and test sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=', blackbox_name)

            ## Classification
            if blackbox_name in ['lg', 'gt', 'nn']:
                ## Creating and training black-box
                BlackBoxConstructor = blackbox_list[blackbox_name]
                blackbox = BlackBoxConstructor(random_state=42)
                blackbox.fit(X_train, Y_train)
                pred_test = blackbox.predict(X_test)
                bb_accuracy_score = accuracy_score(Y_test, pred_test)
                print('blackbox accuracy=', bb_accuracy_score)
                bb_f1_score = f1_score(Y_test, pred_test,average='macro')
                print('blackbox F1-score=', bb_f1_score)
                print('\n')

                ## Generating counterfactuals using MOCF
                ind = 0
                x = X_test[ind]
                label_x = blackbox.predict(x.reshape(1, -1))
                label_cf = int(1 - label_x)  # Desired label
                output_range = [0.7, 1]      # Desired probability range
                output = MOCF(x, blackbox, dataset, output_range=output_range, label_cf=label_cf)

            ## Regression
            elif blackbox_name in ['rfr', 'nnr']:

                ## Creating and training black-box
                BlackBoxConstructor = blackbox_list[blackbox_name]
                blackbox = BlackBoxConstructor(random_state=42)
                blackbox.fit(X_train, Y_train)
                pred_test = blackbox.predict(X_test)
                bb_mae_error = mean_absolute_error(Y_test, pred_test)
                print('blackbox MAE=', bb_mae_error)
                bb_mse_error = mean_squared_error(Y_test, pred_test)
                print('blackbox MSE=', bb_mse_error)
                print('\n')

                def SelectResponseRange(x, blackbox, dataset):
                    q = np.quantile(dataset['y'], q=[0.0, 0.25, 0.75, 1])
                    ranges = [[q[0], q[1]], [q[1], q[2]], [q[2], q[3]]]
                    response_x = blackbox.predict(x.reshape(1, -1))
                    curr_range = -1
                    for i in range(len(ranges)):
                        if ranges[i][0] <= response_x <= ranges[i][1]:
                            curr_range = i
                            break
                    exist_range = list(range(len(ranges)))
                    exist_range.remove(curr_range)
                    cf_range = np.random.choice(exist_range,size=1)
                    return ranges[int(cf_range)]

                ## Generating counterfactuals using MOCF
                ind = 0
                x = X_test[ind]
                output_range = SelectResponseRange(x, blackbox, dataset)    # Desired response range
                output = MOCF(x, blackbox, dataset, output_range=output_range, label_cf=None)

if __name__ == '__main__':
    main()
