from mocf import MOCF
from prepare_datasets import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def main():
    ## Defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    ## Defining the list of data sets
    datsets_list = {
        'breast_cancer': ('breast-cancer.csv', PrepareBreastCancer),
        # 'credit_card_default': ('credit-card-default.csv', PrepareCreditCardDefault),
        # 'adult': ('adult.csv', PrepareAdult),
        # 'boston_house_prices': ('boston-house-prices.csv', PrepareBostonHousePrices)
    }

    ## Defining the list of black-boxes
    blackbox_list = {
        # 'lg': LogisticRegression,
        'gt': GradientBoostingClassifier,
        'nn': MLPClassifier,
        # 'lr': LinearRegression
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
            output = MOCF(x, blackbox, dataset, probability_range=[0.7,1])

if __name__ == '__main__':
    main()
