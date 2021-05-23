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
from evaluate_counterfactuals import evaluateCounterfactuals

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    experiment_size = {
        'adult': (500, 10),
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
                experiment_path + 'care_coherency_preservation_relationship_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_coherency_preservation_relationship_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'care_coherency_preservation_relationship_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = ['Validity',
                      'Validity+Soundness',
                      'Validity+Coherency',
                      'Validity+Soundness+Coherency']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            average = '%s,%s,%s,%s\n' % \
                     ('=average(A3:A1000)', '=average(B3:B1000)', '=average(C3:C1000)', '=average(D3:D1000)')
            eval_results_csv.write(average)
            eval_results_csv.flush()

            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_12.fit(X_train, Y_train)

            # CARE with {validity, coherency} config
            care_config_13 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                   SOUNDNESS=False, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf)
            care_config_13.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency} config
            care_config_123 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf)
            care_config_123.fit(X_train, Y_train)

            # explaining instances from test set
            explained = 0
            for x_ord in X_test:

                explanation_config_1 = care_config_1.explain(x_ord)
                explanation_config_12 = care_config_12.explain(x_ord)
                explanation_config_13 = care_config_13.explain(x_ord)
                explanation_config_123 = care_config_123.explain(x_ord)

                # evaluating counterfactuals based on all objectives results
                toolbox = explanation_config_123['toolbox']
                objective_names = explanation_config_123['objective_names']
                featureScaler = explanation_config_123['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating and recovering counterfactuals of {validity} config
                cfs_ord_config_1, \
                cfs_eval_config_1, \
                x_cfs_ord_config_1, \
                x_cfs_eval_config_1 = evaluateCounterfactuals(x_ord, explanation_config_1['cfs_ord'], dataset,
                                                              predict_fn, predict_proba_fn, task, toolbox,
                                                              objective_names, featureScaler, feature_names)
                relationship = cfs_ord_config_1['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_1['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_1['sex'].to_numpy().astype(int)
                preserved_config_1 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        preserved_config_1 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        preserved_config_1 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        preserved_config_1 += 1
                preserved_config_1 = preserved_config_1 / n_cf

                # evaluating and recovering counterfactuals of {validity, soundness} config
                cfs_ord_config_12, \
                cfs_eval_config_12, \
                x_cfs_ord_config_12, \
                x_cfs_eval_config_12 = evaluateCounterfactuals(x_ord, explanation_config_12['cfs_ord'], dataset,
                                                               predict_fn, predict_proba_fn, task, toolbox,
                                                               objective_names, featureScaler, feature_names)
                relationship = cfs_ord_config_12['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_12['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_12['sex'].to_numpy().astype(int)
                preserved_config_12 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        preserved_config_12 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        preserved_config_12 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        preserved_config_12 += 1
                preserved_config_12 = preserved_config_12 / n_cf

                # evaluating and recovering counterfactuals of {validity, coherency} config
                cfs_ord_config_13, \
                cfs_eval_config_13, \
                x_cfs_ord_config_13, \
                x_cfs_eval_config_13 = evaluateCounterfactuals(x_ord, explanation_config_13['cfs_ord'],
                                                                dataset, predict_fn, predict_proba_fn, task,
                                                                toolbox, objective_names, featureScaler,
                                                                feature_names)
                relationship = cfs_ord_config_13['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_13['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_13['sex'].to_numpy().astype(int)
                preserved_config_13 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        preserved_config_13 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        preserved_config_13 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        preserved_config_13 += 1
                preserved_config_13 = preserved_config_13 / n_cf

                # evaluating and recovering counterfactuals of {validity, soundness, coherency} config
                cfs_ord_config_123, \
                cfs_eval_config_123, \
                x_cfs_ord_config_123, \
                x_cfs_eval_config_123 = evaluateCounterfactuals(x_ord, explanation_config_123['cfs_ord'],
                                                                 dataset, predict_fn, predict_proba_fn, task,
                                                                 toolbox, objective_names, featureScaler,
                                                                 feature_names)
                relationship = cfs_ord_config_123['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_123['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_123['sex'].to_numpy().astype(int)
                preserved_config_123 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        preserved_config_123 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        preserved_config_123 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        preserved_config_123 += 1
                preserved_config_123 = preserved_config_123 / n_cf

                print('\n')
                print('-------------------------------')
                print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                print('\n')
                print(cfs_ord_config_1)
                print(cfs_ord_config_12)
                print(cfs_ord_config_13)
                print(cfs_ord_config_123)
                print('\n')
                print("preserved coherency | Validity: %0.3f - Validity+Soundness: %0.3f - "
                      "Validity+Coherency: %0.3f - Validity+Soundness+Coherency: %0.3f" %
                      (preserved_config_1, preserved_config_12, preserved_config_13, preserved_config_123))
                print('-----------------------------------------------------------------------'
                      '--------------------------------------------------------------')

                # storing the evaluation of the best counterfactual found by methods
                eval_results = np.r_[preserved_config_1, preserved_config_12, preserved_config_13, preserved_config_123]
                eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                eval_results = ','.join(eval_results)
                eval_results_csv.write('%s\n' % (eval_results))
                eval_results_csv.flush()

                explained += 1

                if explained == N:
                    break

            eval_results_csv.close()

if __name__ == '__main__':
    main()