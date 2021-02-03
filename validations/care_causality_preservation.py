import os
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from utils import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from care.care import CARE
from care_explainer import CAREExplainer

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'simple-binomial': ('simple-binomial', PrepareSimpleBinomial, 'classification')
    }

    # defining the list of black-boxes
    blackbox_list = {
        # 'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    experiment_size = {
        'simple-binomial': (200, 5)
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
                predict_proba_fn = lambda x: np.asarray([1 - blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_causality_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_causality_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'care_causality_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = ['Validity', '', '',
                      'Validity+Soundness', '', '',
                      'Validity+Causality', '', '',
                      'Validity+Soundness+Causality', '', '',]
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))

            header = ['n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            eval_results_csv.flush()

            # creating explainer instances
            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, CAUSALITY=False, ACTIONABILITY=False, n_cf=n_cf,
                                 corr_thresh=0.0001, corr_model_score_thresh=0.7)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, CAUSALITY=False, ACTIONABILITY=False, n_cf=n_cf,
                                  corr_thresh=0.0001, corr_model_score_thresh=0.7)
            care_config_12.fit(X_train, Y_train)

            # CARE with {validity, causality} config
            care_config_13 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=False, CAUSALITY=True, ACTIONABILITY=False, n_cf=n_cf,
                                  corr_thresh=0.0001, corr_model_score_thresh=0.7)
            care_config_13.fit(X_train, Y_train)

            # CARE {validity, soundness, causality} config
            care_config_123 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                   SOUNDNESS=True, CAUSALITY=True, ACTIONABILITY=False, n_cf=n_cf,
                                   corr_thresh=0.0001, corr_model_score_thresh=0.7)
            care_config_123.fit(X_train, Y_train)

            # explaining instances from test set
            explained = 0
            config_1_preservation = 0.0
            config_12_preservation = 0.0
            config_13_preservation = 0.0
            config_123_preservation = 0.0
            config_1_n_causes = []
            config_12_n_causes = []
            config_13_n_causes = []
            config_123_n_causes = []
            config_1_n_effects = []
            config_12_n_effects = []
            config_13_n_effects = []
            config_123_n_effects = []
            for x_ord in X_test:

                try:
                    # explaining instance x_ord using CARE with {validity} config
                    config_1_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                    predict_proba_fn, explainer=care_config_1,
                                                    cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    config_1_x_cfs_highlight = config_1_output['x_cfs_highlight']
                    config_1_x_cfs_eval = config_1_output['x_cfs_eval']
                    config_1_cfs_ord = config_1_output['cfs_ord']

                    # explaining instance x_ord using CARE with {validity, soundness} config
                    config_12_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                     predict_proba_fn, explainer=care_config_12,
                                                     cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    config_12_x_cfs_highlight = config_12_output['x_cfs_highlight']
                    config_12_x_cfs_eval = config_12_output['x_cfs_eval']
                    config_12_cfs_ord = config_12_output['cfs_ord']

                    # explaining instance x_ord using CARE with {validity, causality} config
                    config_13_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                     predict_proba_fn, explainer=care_config_13,
                                                     cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    config_13_x_cfs_highlight = config_13_output['x_cfs_highlight']
                    config_13_x_cfs_eval = config_13_output['x_cfs_eval']
                    config_13_cfs_ord = config_13_output['cfs_ord']


                    # explaining instance x_ord using CARE with {validity, soundness, causality} config
                    config_123_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                      predict_proba_fn, explainer=care_config_123,
                                                      cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    config_123_x_cfs_highlight = config_123_output['x_cfs_highlight']
                    config_123_x_cfs_eval = config_123_output['x_cfs_eval']
                    config_123_cfs_ord = config_123_output['cfs_ord']

                    # print counterfactuals and their corresponding objective values
                    print('\n')
                    print('Validity Results')
                    print(pd.concat([config_1_x_cfs_highlight, config_1_x_cfs_eval], axis=1))
                    print('\n')
                    print('Validity+Soundness Results')
                    print(pd.concat([config_12_x_cfs_highlight, config_12_x_cfs_eval], axis=1))
                    print('\n')
                    print('Validity+Causality Results')
                    print(pd.concat([config_13_x_cfs_highlight, config_13_x_cfs_eval], axis=1))
                    print('\n')
                    print('Validity+Soundness+Causality Results')
                    print(pd.concat([config_123_x_cfs_highlight, config_123_x_cfs_eval], axis=1))
                    print('\n')


                    # N.B. third features determined by a monotonically increasing/decreasing function of
                    # first and second features, therefore,
                    # (1st feature and 2nd feature) increase => 3rd feature increases
                    # (1st feature and 2nd feature) decrease => 3rd feature decreases

                    # calculating the number of counterfactuals that preserved causality in
                    # CARE with {validity} config
                    config_1_causes = np.logical_or(
                        np.logical_and(config_1_cfs_ord.iloc[:, 0] > x_ord[0], config_1_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(config_1_cfs_ord.iloc[:, 0] < x_ord[0], config_1_cfs_ord.iloc[:, 1] < x_ord[1]))
                    config_1_effects = np.logical_or(np.logical_and(np.logical_and(config_1_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          config_1_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           config_1_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(config_1_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          config_1_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           config_1_cfs_ord.iloc[:, 2] < x_ord[2]))

                    config_1_n_causes.append(sum(config_1_causes))
                    config_1_n_effects.append(sum(config_1_effects))
                    config_1_preservation = np.mean(config_1_n_effects) / np.mean(config_1_n_causes)

                    # calculating the number of counterfactuals that preserved causality in
                    # CARE with {validity, soundness} config
                    config_12_causes = np.logical_or(np.logical_and(config_12_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          config_12_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(config_12_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          config_12_cfs_ord.iloc[:, 1] < x_ord[1]))
                    config_12_effects = np.logical_or(np.logical_and(np.logical_and(config_12_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          config_12_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           config_12_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(config_12_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          config_12_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           config_12_cfs_ord.iloc[:, 2] < x_ord[2]))

                    config_12_n_causes.append(sum(config_12_causes))
                    config_12_n_effects.append(sum(config_12_effects))
                    config_12_preservation = np.mean(config_12_n_effects) / np.mean(config_12_n_causes)

                    # calculating the number of counterfactuals that preserved causality in
                    # CARE with {validity, causality} config
                    config_13_causes = np.logical_or(np.logical_and(config_13_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          config_13_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(config_13_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          config_13_cfs_ord.iloc[:, 1] < x_ord[1]))
                    config_13_effects = np.logical_or(np.logical_and(np.logical_and(config_13_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          config_13_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           config_13_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(config_13_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          config_13_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           config_13_cfs_ord.iloc[:, 2] < x_ord[2]))

                    config_13_n_causes.append(sum(config_13_causes))
                    config_13_n_effects.append(sum(config_13_effects))
                    config_13_preservation = np.mean(config_13_n_effects) / np.mean(config_13_n_causes)


                    # calculating the number of counterfactuals that preserved causality in
                    # CARE with {validity, soundness, causality} config
                    config_123_causes = np.logical_or(
                        np.logical_and(config_123_cfs_ord.iloc[:, 0] > x_ord[0], config_123_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(config_123_cfs_ord.iloc[:, 0] < x_ord[0], config_123_cfs_ord.iloc[:, 1] < x_ord[1]))
                    config_123_effects = np.logical_or(np.logical_and(np.logical_and(config_123_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          config_123_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           config_123_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(config_123_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          config_123_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           config_123_cfs_ord.iloc[:, 2] < x_ord[2]))

                    config_123_n_causes.append(sum(config_123_causes))
                    config_123_n_effects.append(sum(config_123_effects))
                    config_123_preservation = np.mean(config_123_n_effects) / np.mean(config_123_n_causes)

                    explained += 1

                    print('\n')
                    print('-----------------------------------------------------------------------')
                    print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print("preserved causality | Validity: %0.3f - Validity+Soundness: %0.3f - "
                          "Validity+Causality: %0.3f - Validity+Soundness+Causality: %0.3f" %
                          (config_1_preservation, config_12_preservation, config_13_preservation, config_123_preservation))
                    print('-----------------------------------------------------------------------')

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[np.mean(config_1_n_causes), np.mean(config_1_n_effects), config_1_preservation,
                                         np.mean(config_12_n_causes), np.mean(config_12_n_effects), config_12_preservation,
                                         np.mean(config_13_n_causes), np.mean(config_13_n_effects), config_13_preservation,
                                         np.mean(config_123_n_causes), np.mean(config_123_n_effects), config_123_preservation]
                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                except Exception:
                    pass

                if explained == N:
                    break

            eval_results_csv.close()

if __name__ == '__main__':
    main()
