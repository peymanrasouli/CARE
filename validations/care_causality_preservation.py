import os
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from user_preferences import userPreferences
from care.care import CARE
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals
from care.causality import causality
from utils import ord2theta
from itertools import permutations

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'iris': ('iris-sklearn', PrepareIris, 'classification')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'gb-c': GradientBoostingClassifier,
    }

    experiment_size = {
        'iris': (30, 5),
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name,usage='causality_validation')

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

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_causality_preservation_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_causality_preservation_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            action_series_results_csv = open(
                experiment_path + 'care_causality_preservation_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            # creating an instance of CARE explainer
            explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                             sound=True, causality=True, actionable=False, corr_model_score_thresh=0.2, n_cf=n_cf)

            # fitting the explainer on the training data
            explainer.fit(X_train, Y_train)

            # explaining test samples
            explained = 0
            for x_ord in X_test:

                # generating counterfactuals
                explanations = explainer.explain(x_ord, cf_class='strange')

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


                # print counterfactuals and their corresponding objective values
                print('\n')
                print(x_cfs_highlight)
                print('\n')
                print(x_cfs_eval)
                print('\n')

                # storing the counterfactuals
                action_series_results = pd.concat([x_cfs_highlight, x_cfs_eval], axis=1)
                action_series_results.to_csv(action_series_results_csv)
                action_series_results_csv.write('\n')
                action_series_results_csv.flush()

                explained += 1

                print('\n')
                print('-----------------------------------------------------------------------')
                print("%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                print('-----------------------------------------------------------------------')

                if explained == N:
                    break

            print('\n')
            print('Done!')

if __name__ == '__main__':
    main()


