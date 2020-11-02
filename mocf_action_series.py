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
from mocf import MOCF
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals
from correlation import correlation
from utils import ord2theta
from itertools import permutations

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

            # creating an instance of MOCF explainer
            n_cf = 10
            explainer = MOCF(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                             soundCF=True, feasibleAR=False, n_cf=n_cf)

            # fitting the explainer on the training data
            explainer.fit(X_train, Y_train)

            for x_ord in X_test:

                # set user preferences
                user_preferences = userPreferences(dataset, x_ord)

                # generating counterfactuals
                explanations = explainer.explain(x_ord, cf_class='opposite', cf_quantile='neighbor',
                                                 probability_thresh=0.5, user_preferences=user_preferences)

                # extracting results
                cfs_ord = explanations['cfs_ord']
                toolbox = explanations['toolbox']
                objective_names = explanations['objective_names']
                featureScaler = explanations['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counterfactuals
                cfs_ord, \
                cfs_eval, \
                x_cfs_ord, \
                x_cfs_eval = evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                                                     toolbox, objective_names, featureScaler, feature_names)

                # recovering counterfactuals in original format
                x_org, \
                cfs_org, \
                x_cfs_org, \
                x_cfs_highlight = recoverOriginals(x_ord, cfs_ord, dataset, feature_names)

                # action series
                cfs_action_series = []
                for i in range(n_cf):
                    changed_features = np.where(cfs_ord.to_numpy()[i, :] != x_ord)[0]
                    orders = permutations(list(changed_features), len(changed_features))
                    order_cost = {}
                    for o in list(orders):
                        corr_cost = []
                        cf_ord = x_ord.copy()
                        for f in list(o):
                            cf_ord[f] = cfs_ord.iloc[i, f]
                            cf_theta = ord2theta(cf_ord, explainer.featureScaler)
                            corr = correlation(x_ord, cf_ord, cf_theta, dataset['feature_width'],
                                               dataset['continuous_indices'],
                                               dataset['discrete_indices'], explainer.correlationModel)
                            corr_cost.append(corr)
                        order_cost[o] = np.mean(corr_cost)
                    cfs_action_series.append(order_cost)

                best_series = [None] * n_cf
                worst_series  = [None] * n_cf
                for i, d in enumerate(cfs_action_series):
                    best_val = np.inf
                    worst_val = 0
                    for key, val in d.items():
                        if val < best_val:
                            best_series[i] = {key: val}
                            best_val = val
                        if val > worst_val:
                            worst_series[i] = {key: val}
                            worst_val = val

                # print n best counterfactuals and their corresponding objective values
                print('\n')
                print(x_cfs_highlight.head(n= n_cf + 1))
                print('\n')
                print(x_cfs_eval.head(n= n_cf + 1))
                print('\n')
                print('Best action series:')
                for i, bs in enumerate(best_series):
                    for key, val in bs.items():
                        print('cf_'+str(i)+':', [dataset['feature_names'][f] for f in key], '| cost:', val)
                print('\n')
                print('Worst action series:')
                for i, ws in enumerate(worst_series):
                    if ws == None:
                        print('cf_'+str(i)+':', 'not available!')
                    else:
                        for key, val in ws.items():
                            print('cf_'+str(i)+':', [dataset['feature_names'][f] for f in key], '| cost:', val)

            print('\n')
            print('Done!')

if __name__ == '__main__':
    main()


