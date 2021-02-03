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
                      'Soundness', '', '',
                      'Causality', '', '',
                      'Soundness+Causality', '', '',]
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
            # CARE validity
            validity_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=False, CAUSALITY=False, ACTIONABILITY=False, n_cf=n_cf,
                                  corr_thresh=0.0001, corr_model_score_thresh=0.7)
            validity_explainer.fit(X_train, Y_train)

            # CARE soundness
            soundness_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, CAUSALITY=False, ACTIONABILITY=False, n_cf=n_cf,
                                  corr_thresh=0.0001, corr_model_score_thresh=0.7)
            soundness_explainer.fit(X_train, Y_train)

            # CARE causality
            causality_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                       SOUNDNESS=False, CAUSALITY=True, ACTIONABILITY=False, n_cf=n_cf,
                                       corr_thresh=0.0001, corr_model_score_thresh=0.7)
            causality_explainer.fit(X_train, Y_train)

            # CARE soundness+causality
            soundness_causality_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                             SOUNDNESS=True, CAUSALITY=True, ACTIONABILITY=False, n_cf=n_cf,
                                             corr_thresh=0.0001, corr_model_score_thresh=0.7)
            soundness_causality_explainer.fit(X_train, Y_train)

            # explaining instances from test set
            explained = 0
            validity_preservation = 0.0
            soundness_preservation = 0.0
            causality_preservation = 0.0
            soundness_causality_preservation = 0.0
            validity_n_causes = []
            soundness_n_causes = []
            causality_n_causes = []
            soundness_causality_n_causes = []
            validity_n_effects = []
            soundness_n_effects = []
            causality_n_effects = []
            soundness_causality_n_effects = []
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE validity
                    validity_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=validity_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    validity_x_cfs_highlight = validity_output['x_cfs_highlight']
                    validity_x_cfs_eval = validity_output['x_cfs_eval']
                    validity_cfs_ord = validity_output['cfs_ord']

                    # explain instance x_ord using CARE soundness
                    soundness_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=soundness_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    soundness_x_cfs_highlight = soundness_output['x_cfs_highlight']
                    soundness_x_cfs_eval = soundness_output['x_cfs_eval']
                    soundness_cfs_ord = soundness_output['cfs_ord']

                    # explain instance x_ord using CARE causality
                    causality_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                     predict_proba_fn, explainer=causality_explainer,
                                                     cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    causality_x_cfs_highlight = causality_output['x_cfs_highlight']
                    causality_x_cfs_eval = causality_output['x_cfs_eval']
                    causality_cfs_ord = causality_output['cfs_ord']


                    # explain instance x_ord using CARE soundness_causality
                    soundness_causality_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=soundness_causality_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    soundness_causality_x_cfs_highlight = soundness_causality_output['x_cfs_highlight']
                    soundness_causality_x_cfs_eval = soundness_causality_output['x_cfs_eval']
                    soundness_causality_cfs_ord = soundness_causality_output['cfs_ord']

                    # print counterfactuals and their corresponding objective values
                    print('\n')
                    print('Validity Results')
                    print(pd.concat([validity_x_cfs_highlight, validity_x_cfs_eval], axis=1))
                    print('\n')
                    print('Soundness Results')
                    print(pd.concat([soundness_x_cfs_highlight, soundness_x_cfs_eval], axis=1))
                    print('\n')
                    print('Causality Results')
                    print(pd.concat([causality_x_cfs_highlight, causality_x_cfs_eval], axis=1))
                    print('\n')
                    print('Soundness+Causality Results')
                    print(pd.concat([soundness_causality_x_cfs_highlight, soundness_causality_x_cfs_eval], axis=1))
                    print('\n')


                    # N.B. third features determined by a monotonically increasing/decreasing function of
                    # first and second features, therefore,
                    # (1st feature and 2nd feature) increase => 3rd feature increases
                    # (1st feature and 2nd feature) decrease => 3rd feature decreases

                    # calculating the number of counterfactuals that preserved causality in CARE validity method
                    validity_causes = np.logical_or(
                        np.logical_and(validity_cfs_ord.iloc[:, 0] > x_ord[0], validity_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(validity_cfs_ord.iloc[:, 0] < x_ord[0], validity_cfs_ord.iloc[:, 1] < x_ord[1]))
                    validity_effects = np.logical_or(np.logical_and(np.logical_and(validity_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          validity_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           validity_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(validity_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          validity_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           validity_cfs_ord.iloc[:, 2] < x_ord[2]))

                    validity_n_causes.append(sum(validity_causes))
                    validity_n_effects.append(sum(validity_effects))
                    validity_preservation = np.mean(validity_n_effects) / np.mean(validity_n_causes)

                    # calculating the number of counterfactuals that preserved causality in CARE soundness method
                    soundness_causes = np.logical_or(np.logical_and(soundness_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          soundness_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(soundness_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          soundness_cfs_ord.iloc[:, 1] < x_ord[1]))
                    soundness_effects = np.logical_or(np.logical_and(np.logical_and(soundness_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          soundness_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           soundness_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(soundness_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          soundness_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           soundness_cfs_ord.iloc[:, 2] < x_ord[2]))

                    soundness_n_causes.append(sum(soundness_causes))
                    soundness_n_effects.append(sum(soundness_effects))
                    soundness_preservation = np.mean(soundness_n_effects) / np.mean(soundness_n_causes)

                    # calculating the number of counterfactuals that preserved causality in CARE causality method
                    causality_causes = np.logical_or(np.logical_and(causality_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(causality_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          causality_cfs_ord.iloc[:, 1] < x_ord[1]))
                    causality_effects = np.logical_or(np.logical_and(np.logical_and(causality_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           causality_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(causality_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          causality_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           causality_cfs_ord.iloc[:, 2] < x_ord[2]))

                    causality_n_causes.append(sum(causality_causes))
                    causality_n_effects.append(sum(causality_effects))
                    causality_preservation = np.mean(causality_n_effects) / np.mean(causality_n_causes)


                    # calculating the number of counterfactuals that preserved causality in CARE soundness+causality method
                    soundness_causality_causes = np.logical_or(
                        np.logical_and(soundness_causality_cfs_ord.iloc[:, 0] > x_ord[0], soundness_causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(soundness_causality_cfs_ord.iloc[:, 0] < x_ord[0], soundness_causality_cfs_ord.iloc[:, 1] < x_ord[1]))
                    soundness_causality_effects = np.logical_or(np.logical_and(np.logical_and(soundness_causality_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          soundness_causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           soundness_causality_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(soundness_causality_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          soundness_causality_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           soundness_causality_cfs_ord.iloc[:, 2] < x_ord[2]))

                    soundness_causality_n_causes.append(sum(soundness_causality_causes))
                    soundness_causality_n_effects.append(sum(soundness_causality_effects))
                    soundness_causality_preservation = np.mean(soundness_causality_n_effects) / np.mean(soundness_causality_n_causes)

                    explained += 1

                    print('\n')
                    print('-----------------------------------------------------------------------')
                    print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print("preserved causality | Validity: %0.3f - Soundness: %0.3f - Causality: %0.3f - Soundness+Causality: %0.3f" %
                          (validity_preservation, soundness_preservation, causality_preservation, soundness_causality_preservation))
                    print('-----------------------------------------------------------------------')

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[np.mean(validity_n_causes), np.mean(validity_n_effects), validity_preservation,
                                         np.mean(soundness_n_causes), np.mean(soundness_n_effects), soundness_preservation,
                                         np.mean(causality_n_causes), np.mean(causality_n_effects), causality_preservation,
                                         np.mean(soundness_causality_n_causes), np.mean(soundness_causality_n_effects), soundness_causality_preservation]
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
