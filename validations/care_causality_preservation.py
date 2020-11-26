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
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    experiment_size = {
        'simple-binomial': (1000, 1)
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
                experiment_path + 'care_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf))
            eval_results_csv = open(
                experiment_path + 'care_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf), 'a')

            header = ['Valid', '', '',
                      'Sound', '', '',
                      'Sound+Causality', '', '',]
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))

            header = ['n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            eval_results_csv.flush()

            # creating explainer instances
            # CARE valid
            valid_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  sound=False, causality=False, actionable=False, corr_model_score_thresh=0.7, n_cf=n_cf)
            valid_explainer.fit(X_train, Y_train)

            # CARE sound
            sound_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  sound=True, causality=False, actionable=False, corr_model_score_thresh=0.7, n_cf=n_cf)
            sound_explainer.fit(X_train, Y_train)

            # CARE sound+causality
            sound_causality_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  sound=True, causality=True, actionable=False, corr_model_score_thresh=0.7, n_cf=n_cf)
            sound_causality_explainer.fit(X_train, Y_train)

            # explaining instances from test set
            explained = 0
            valid_preservation = []
            sound_preservation = []
            sound_causality_preservation = []
            valid_n_causes = []
            sound_n_causes = []
            sound_causality_n_causes = []
            valid_n_effects = []
            sound_n_effects = []
            sound_causality_n_effects = []
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE valid
                    valid_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=valid_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    valid_x_cfs_highlight = valid_output['x_cfs_highlight']
                    valid_x_cfs_eval = valid_output['x_cfs_eval']
                    valid_cfs_ord = valid_output['cfs_ord']

                    # explain instance x_ord using CARE sound
                    sound_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=sound_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    sound_x_cfs_highlight = sound_output['x_cfs_highlight']
                    sound_x_cfs_eval = sound_output['x_cfs_eval']
                    sound_cfs_ord = sound_output['cfs_ord']

                    # explain instance x_ord using CARE sound_causality
                    sound_causality_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=sound_causality_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    sound_causality_x_cfs_highlight = sound_causality_output['x_cfs_highlight']
                    sound_causality_x_cfs_eval = sound_causality_output['x_cfs_eval']
                    sound_causality_cfs_ord = sound_causality_output['cfs_ord']

                    # print counterfactuals and their corresponding objective values
                    print('\n')
                    print('Valid Results')
                    print(pd.concat([valid_x_cfs_highlight, valid_x_cfs_eval], axis=1))
                    print('\n')
                    print('Sound Results')
                    print(pd.concat([sound_x_cfs_highlight, sound_x_cfs_eval], axis=1))
                    print('\n')
                    print('Sound+Causality Results')
                    print(pd.concat([sound_causality_x_cfs_highlight, sound_causality_x_cfs_eval], axis=1))
                    print('\n')


                    # N.B. third features determined by a monotonically increasing/decreasing function of
                    # first and second features, therefore,
                    # (1st feature and 2nd feature) increase => 3rd feature increases
                    # (1st feature and 2nd feature) decrease => 3rd feature decreases

                    # calculating the number of counterfactuals that preserved causality in CARE valid method
                    causes = np.logical_or(
                        np.logical_and(valid_cfs_ord.iloc[:, 0] > x_ord[0], valid_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(valid_cfs_ord.iloc[:, 0] < x_ord[0], valid_cfs_ord.iloc[:, 1] < x_ord[1]))
                    effects = np.logical_or(np.logical_and(np.logical_and(valid_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          valid_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           valid_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(valid_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          valid_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           valid_cfs_ord.iloc[:, 2] < x_ord[2]))
                    if sum(causes) == 0:
                        pass
                    else:
                        valid_n_causes.append(causes)
                        valid_n_effects.append(effects)
                        valid_preservation.append(sum(effects) / sum(causes))

                    # calculating the number of counterfactuals that preserved causality in sound and CARE sound method
                    causes = np.logical_or(np.logical_and(sound_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          sound_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(sound_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          sound_cfs_ord.iloc[:, 1] < x_ord[1]))
                    effects = np.logical_or(np.logical_and(np.logical_and(sound_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          sound_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           sound_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(sound_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          sound_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           sound_cfs_ord.iloc[:, 2] < x_ord[2]))
                    if sum(causes) == 0:
                        pass
                    else:
                        sound_n_causes.append(causes)
                        sound_n_effects.append(effects)
                        sound_preservation.append(sum(effects) / sum(causes))


                    # calculating the number of counterfactuals that preserved causality in CARE sound+causality method
                    causes = np.logical_or(
                        np.logical_and(sound_causality_cfs_ord.iloc[:, 0] > x_ord[0], sound_causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(sound_causality_cfs_ord.iloc[:, 0] < x_ord[0], sound_causality_cfs_ord.iloc[:, 1] < x_ord[1]))
                    effects = np.logical_or(np.logical_and(np.logical_and(sound_causality_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          sound_causality_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           sound_causality_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(sound_causality_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          sound_causality_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           sound_causality_cfs_ord.iloc[:, 2] < x_ord[2]))
                    if sum(causes) == 0:
                        pass
                    else:
                        sound_causality_n_causes.append(causes)
                        sound_causality_n_effects.append(effects)
                        sound_causality_preservation.append(sum(effects) / sum(causes))


                    explained += 1

                    print('\n')
                    print('-----------------------------------------------------------------------')
                    print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print("preserved causality | Valid: %0.3f - Sound: %0.3f - Sound+Causality: %0.3f" %
                          (np.mean(valid_preservation), np.mean(sound_preservation), np.mean(sound_causality_preservation)))
                    print('-----------------------------------------------------------------------')

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[np.sum(valid_n_causes), np.sum(valid_n_effects), np.mean(valid_preservation),
                                         np.sum(sound_n_causes), np.sum(sound_n_effects), np.mean(sound_preservation),
                                         np.sum(sound_causality_n_causes), np.sum(sound_causality_n_effects), np.mean(sound_causality_preservation)]
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
