import warnings
warnings.filterwarnings("ignore")
import os
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

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier,
        # 'nn-r': MLPRegressor,
        # 'gb-r': GradientBoostingRegressor
    }

    experiment_size = {
        'adult': (500, 5),
        'credit-card_default': (500, 5),
        'heart-disease': (50, 5),
        'boston-house-prices': (100, 5)
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

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating an instance of MOCF explainer for  soundCF=False and feasibleAR=False
            explainer_base = MOCF(dataset, task=task, predict_fn=predict_fn,
                                  predict_proba_fn=predict_proba_fn,
                                  soundCF=False, feasibleAR=False, n_cf=n_cf)
            explainer_base.fit(X_train, Y_train)

            # creating an instance of MOCF explainer for  soundCF=True and feasibleAR=False
            explainer_sound = MOCF(dataset, task=task, predict_fn=predict_fn,
                                   predict_proba_fn=predict_proba_fn,
                                   soundCF=True, feasibleAR=False, n_cf=n_cf)
            explainer_sound.fit(X_train, Y_train)

            # creating an instance of MOCF explainer for  soundCF=False and feasibleAR=True
            explainer_feasible = MOCF(dataset, task=task, predict_fn=predict_fn,
                                   predict_proba_fn=predict_proba_fn,
                                   soundCF=False, feasibleAR=True, n_cf=n_cf)
            explainer_feasible.fit(X_train, Y_train)

            # creating an instance of MOCF explainer for  soundCF=True and feasibleAR=True
            explainer_sound_feasible = MOCF(dataset, task=task, predict_fn=predict_fn,
                                            predict_proba_fn=predict_proba_fn,
                                            soundCF=True, feasibleAR=True, n_cf=n_cf)
            explainer_sound_feasible.fit(X_train, Y_train)

            ################################### Explaining test samples #########################################

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            cfs_results_csv = open(
                experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            feature_space = ['' for _ in range(X_train.shape[1] - 1 + 9)]
            header = ['', 'Base']
            header += feature_space
            header += ['Sound']
            header += feature_space
            header += ['Feasible']
            header += feature_space
            header += ['Sound & Feasible']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('Base', '', '', '', '', '', '', '', '',
                      'Sound', '', '', '', '', '', '', '', '',
                      'Feasible', '', '', '', '', '', '', '', '',
                      'Sound & Feasible', '', '', '', '', '', '', '', '')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                      'sparsity', 'validity', 'diversity',
                      'prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                      'sparsity', 'validity', 'diversity',
                      'prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                      'sparsity', 'validity', 'diversity',
                      'prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                      'sparsity', 'validity', 'diversity')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=average(A5:A1000)', '=average(B5:B1000)',
                      '=average(C5:C1000)', '=average(D5:D1000)',
                      '=average(E5:E1000)', '=average(F5:F1000)',
                      '=average(G5:G1000)', '=average(H5:H1000)',
                      '=average(I5:I1000)', '=average(J5:J1000)',
                      '=average(K5:K1000)', '=average(L5:L1000)',
                      '=average(M5:M1000)', '=average(N5:N1000)',
                      '=average(O5:O1000)', '=average(P5:P1000)',
                      '=average(Q5:Q1000)', '=average(R5:R1000)',
                      '=average(S5:S1000)', '=average(T5:T1000)',
                      '=average(U5:U1000)', '=average(V5:V1000)',
                      '=average(W5:W1000)', '=average(X5:X1000)',
                      '=average(Y5:Y1000)', '=average(Z5:Z1000)',
                      '=average(AA5:AA1000)', '=average(AB5:AB1000)',
                      '=average(AC5:AC1000)', '=average(AD5:AD1000)',
                      '=average(AE5:AE1000)', '=average(AF5:AF1000)',
                      '=average(AG5:AG1000)', '=average(AH5:AH1000)',
                      '=average(AI5:AI1000)', '=average(AJ5:AJ1000)',)
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=stdev(A5:A1000)', '=stdev(B5:B1000)',
                      '=stdev(C5:C1000)', '=stdev(D5:D1000)',
                      '=stdev(E5:E1000)', '=stdev(F5:F1000)',
                      '=stdev(G5:G1000)', '=stdev(H5:H1000)',
                      '=stdev(I5:I1000)', '=stdev(J5:J1000)',
                      '=stdev(K5:K1000)', '=stdev(L5:L1000)',
                      '=stdev(M5:M1000)', '=stdev(N5:N1000)',
                      '=stdev(O5:O1000)', '=stdev(P5:P1000)',
                      '=stdev(Q5:Q1000)', '=stdev(R5:R1000)',
                      '=stdev(S5:S1000)', '=stdev(T5:T1000)',
                      '=stdev(U5:U1000)', '=stdev(V5:V1000)',
                      '=stdev(W5:W1000)', '=stdev(X5:X1000)',
                      '=stdev(Y5:Y1000)', '=stdev(Z5:Z1000)',
                      '=stdev(AA5:AA1000)', '=stdev(AB5:AB1000)',
                      '=stdev(AC5:AC1000)', '=stdev(AD5:AD1000)',
                      '=stdev(AE5:AE1000)', '=stdev(AF5:AF1000)',
                      '=stdev(AG5:AG1000)', '=stdev(AH5:AH1000)',
                      '=stdev(AI5:AI1000)', '=stdev(AJ5:AJ1000)',)
            eval_results_csv.write(header)
            eval_results_csv.flush()

            # explaining instances from test set
            explained = 0
            for x_ord in X_test:

                try:
                    explanation_base = explainer_base.explain(x_ord)
                    explanation_sound = explainer_sound.explain(x_ord)
                    user_preferences = userPreferences(dataset, x_ord)
                    explanation_feasible = explainer_feasible.explain(x_ord, user_preferences=user_preferences)
                    explanation_sound_feasible= explainer_sound_feasible.explain(x_ord, user_preferences=user_preferences)

                    # evaluating counterfactuals based on all objectives results
                    toolbox = explanation_sound_feasible['toolbox']
                    objective_names = explanation_sound_feasible['objective_names']
                    featureScaler = explanation_sound_feasible['featureScaler']
                    feature_names = dataset['feature_names']

                    # evaluating and recovering counterfactuals of base method
                    cfs_ord_base, \
                    cfs_eval_base, \
                    x_cfs_ord_base, \
                    x_cfs_eval_base = evaluateCounterfactuals(x_ord, explanation_base['cfs_ord'], dataset,
                                                              predict_fn, predict_proba_fn, task, toolbox,
                                                              objective_names, featureScaler, feature_names)
                    x_org_base, \
                    cfs_org_base, \
                    x_cfs_org_base, \
                    x_cfs_highlight_base = recoverOriginals(x_ord, cfs_ord_base, dataset, feature_names)


                    # evaluating and recovering counterfactuals of sound method
                    cfs_ord_sound, \
                    cfs_eval_sound, \
                    x_cfs_ord_sound, \
                    x_cfs_eval_sound = evaluateCounterfactuals(x_ord, explanation_sound['cfs_ord'], dataset,
                                                               predict_fn, predict_proba_fn, task, toolbox,
                                                               objective_names, featureScaler, feature_names)
                    x_org_sound, \
                    cfs_org_sound, \
                    x_cfs_org_sound, \
                    x_cfs_highlight_sound = recoverOriginals(x_ord, cfs_ord_sound, dataset, feature_names)


                    # evaluating and recovering counterfactuals of feasible method
                    cfs_ord_feasible, \
                    cfs_eval_feasible, \
                    x_cfs_ord_feasible, \
                    x_cfs_eval_feasible = evaluateCounterfactuals(x_ord, explanation_feasible['cfs_ord'],
                                                                        dataset, predict_fn, predict_proba_fn, task,
                                                                        toolbox, objective_names, featureScaler,
                                                                        feature_names)
                    x_org_feasible, \
                    cfs_org_feasible, \
                    x_cfs_org_feasible, \
                    x_cfs_highlight_feasible = recoverOriginals(x_ord, cfs_ord_feasible, dataset, feature_names)


                    # evaluating and recovering counterfactuals of sound and feasible method
                    cfs_ord_sound_feasible, \
                    cfs_eval_sound_feasible, \
                    x_cfs_ord_sound_feasible, \
                    x_cfs_eval_sound_feasible = evaluateCounterfactuals(x_ord, explanation_sound_feasible['cfs_ord'],
                                                                        dataset, predict_fn, predict_proba_fn, task,
                                                                        toolbox, objective_names, featureScaler,
                                                                        feature_names)
                    x_org_sound_feasible, \
                    cfs_org_sound_feasible, \
                    x_cfs_org_sound_feasible, \
                    x_cfs_highlight_sound_feasible = recoverOriginals(x_ord, cfs_ord_sound_feasible, dataset, feature_names)

                    # finding the index of the best counterfactual in the output of every method
                    idx_best_base = (np.where((x_cfs_ord_base ==
                                               explanation_base['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_sound = (np.where((x_cfs_ord_sound ==
                                                explanation_sound['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_feasible = (np.where((x_cfs_ord_feasible ==
                                                   explanation_feasible['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_sound_feasible = (np.where((x_cfs_ord_sound_feasible ==
                                                         explanation_sound_feasible['best_cf_ord']).all(axis=1)==True))[0][0]

                    # storing the best counterfactual found by methods
                    cfs_results = pd.concat([x_cfs_highlight_base.iloc[[0,idx_best_base]],
                                             x_cfs_eval_base.iloc[[0,idx_best_base]],
                                             x_cfs_highlight_sound.iloc[[0,idx_best_sound]],
                                             x_cfs_eval_sound.iloc[[0,idx_best_sound]],
                                             x_cfs_highlight_feasible.iloc[[0,idx_best_feasible]],
                                             x_cfs_eval_feasible.iloc[[0,idx_best_feasible]],
                                             x_cfs_highlight_sound_feasible.iloc[[0,idx_best_sound_feasible]],
                                             x_cfs_eval_sound_feasible.iloc[[0,idx_best_sound_feasible]]], axis=1)
                    cfs_results.to_csv(cfs_results_csv)
                    cfs_results_csv.write('\n')
                    cfs_results_csv.flush()

                    # storing the evaluation of the best counterfactual found by methods
                    ind = 1 if task == 'regression' else 2
                    eval_results = np.r_[x_cfs_eval_base.iloc[idx_best_base, :-ind],
                                         x_cfs_eval_sound.iloc[idx_best_sound, :-ind],
                                         x_cfs_eval_feasible.iloc[idx_best_feasible, :-ind],
                                         x_cfs_eval_sound_feasible.iloc[idx_best_sound_feasible, :-ind]]
                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                    explained += 1

                except Exception:
                    pass

                if explained == N:
                    break

            cfs_results_csv.close()
            eval_results_csv.close()

            print('Done!')

if __name__ == '__main__':
    main()
