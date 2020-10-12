import warnings
warnings.filterwarnings("ignore")
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, KerasNeuralNetwork
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from user_preferences import userPreferences
from mocf_explainer import mocfExplainer

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
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # Classification
            if task is 'classification':
                ## User preferences

                explainer = mocfExplainer(dataset, task=task, predict_fn=predict_fn,predict_proba_fn=predict_proba_fn, soundCF=False)
                explainer.fit(X_train, Y_train)
                ind = 1
                x_ord = X_test[ind]
                preferences = userPreferences(dataset, x_ord)
                explanation = explainer.explain(x_ord, cf_class='opposite', probability_thresh=0.6, user_preferences=preferences)
                print('Done!')

            # Regression
            elif task is 'regression':
                explainer = mocfExplainer(dataset, task=task, predict_fn=predict_fn, soundCF=True)
                explainer.fit(X_train, Y_train)
                ind = 0
                x_ord = X_test[ind]
                preferences = userPreferences(dataset, x_ord)
                explanation = explainer.explain(x_ord, cf_quantile='neighbor', user_preferences=preferences)
                print('Done!')

if __name__ == '__main__':
    main()
