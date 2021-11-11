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
from create_model import CreateModel, MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from user_preferences import userPreferences
from care.care import CARE
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPAS, 'classification'),
        'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        'heloc': ('heloc_dataset_v1.csv', PrepareHELOC, 'classification'),
        'wine': ('wine-sklearn', PrepareWine, 'classification'),
        'iris': ('iris-sklearn', PrepareIris, 'classification'),
        # 'diabetes': ('diabetes-sklearn', PrepareDiabetes, 'regression'),
        # 'california-housing': ('california-housing-sklearn', PrepareCaliforniaHousing, 'regression')
    }

    # defining the list of black-boxes
    blackbox_list = {
        # 'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier,
        # 'nn-r': MLPRegressor,
        # 'gb-r': GradientBoostingRegressor
    }

    # defining number of samples N and number of counterfactuals generated for every instance n_cf
    experiment_size = {
        'adult': (500, 10),
        'compas-scores-two-years': (500, 10),
        'credit-card-default': (500, 10),
        'heloc': (500,10),
        'wine': (30, 10),
        'iris': (30, 10),
        'diabetes': (80, 10),
        'california-housing': (400, 10)
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

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_module_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_module_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            cfs_results_csv = open(
                experiment_path + 'care_module_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            n_out = int(task == 'classification') + 1
            n_metrics = 12
            feature_space = ['' for _ in range(X_train.shape[1] - 1 + n_metrics + n_out)]
            header = ['', 'Validity']
            header += feature_space
            header += ['Validity+Soundness']
            header += feature_space
            header += ['Validity+Soundness+Coherency']
            header += feature_space
            header += ['Validity+Soundness+Coherency+Actionability']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_module_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_module_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'care_module_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('Validity', '', '', '', '', '', '', '', '', '', '', '',
                      'Validity+Soundness', '', '', '', '', '', '', '', '', '', '', '',
                      'Validity+Soundness+Coherency', '', '', '', '', '', '', '', '', '', '', '',
                      'Validity+Soundness+Coherency+Actionability', '', '', '', '', '', '', '', '', '', '', '')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('Outcome', 'Actionability', 'Coherency', 'Proximity', 'Connectedness', 'Distance',
                      'Sparsity', 'i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity',
                      'Outcome', 'Actionability', 'Coherency', 'Proximity', 'Connectedness', 'Distance',
                      'Sparsity', 'i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity',
                      'Outcome', 'Actionability', 'Coherency', 'Proximity', 'Connectedness', 'Distance',
                      'Sparsity', 'i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity',
                      'Outcome', 'Actionability', 'Coherency', 'Proximity', 'Connectedness', 'Distance',
                      'Sparsity', 'i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=average(A5:A1000)', '=average(B5:B1000)', '=average(C5:C1000)', '=average(D5:D1000)',
                      '=average(E5:E1000)', '=average(F5:F1000)', '=average(G5:G1000)', '=average(H5:H1000)',
                      '=average(I5:I1000)', '=average(J5:J1000)', '=average(K5:K1000)', '=average(L5:L1000)',
                      '=average(M5:M1000)', '=average(N5:N1000)', '=average(O5:O1000)', '=average(P5:P1000)',
                      '=average(Q5:Q1000)', '=average(R5:R1000)', '=average(S5:S1000)', '=average(T5:T1000)',
                      '=average(U5:U1000)', '=average(V5:V1000)', '=average(W5:W1000)', '=average(X5:X1000)',
                      '=average(Y5:Y1000)', '=average(Z5:Z1000)', '=average(AA5:AA1000)', '=average(AB5:AB1000)',
                      '=average(AC5:AC1000)', '=average(AD5:AD1000)', '=average(AE5:AE1000)', '=average(AF5:AF1000)',
                      '=average(AG5:AG1000)', '=average(AH5:AH1000)', '=average(AI5:AI1000)', '=average(AJ5:AJ1000)',
                      '=average(AK5:AK1000)', '=average(AL5:AL1000)', '=average(AM5:AM1000)', '=average(AN5:AN1000)',
                      '=average(AO5:AO1000)', '=average(AP5:AP1000)', '=average(AQ5:AQ1000)', '=average(AR5:AR1000)',
                      '=average(AS5:AS1000)', '=average(AT5:AT1000)', '=average(AU5:AU1000)', '=average(AV5:AV1000)',)
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=stdev(A5:A1000)', '=stdev(B5:B1000)', '=stdev(C5:C1000)', '=stdev(D5:D1000)',
                      '=stdev(E5:E1000)', '=stdev(F5:F1000)', '=stdev(G5:G1000)', '=stdev(H5:H1000)',
                      '=stdev(I5:I1000)', '=stdev(J5:J1000)', '=stdev(K5:K1000)', '=stdev(L5:L1000)',
                      '=stdev(M5:M1000)', '=stdev(N5:N1000)', '=stdev(O5:O1000)', '=stdev(P5:P1000)',
                      '=stdev(Q5:Q1000)', '=stdev(R5:R1000)', '=stdev(S5:S1000)', '=stdev(T5:T1000)',
                      '=stdev(U5:U1000)', '=stdev(V5:V1000)', '=stdev(W5:W1000)', '=stdev(X5:X1000)',
                      '=stdev(Y5:Y1000)', '=stdev(Z5:Z1000)', '=stdev(AA5:AA1000)', '=stdev(AB5:AB1000)',
                      '=stdev(AC5:AC1000)', '=stdev(AD5:AD1000)', '=stdev(AE5:AE1000)', '=stdev(AF5:AF1000)',
                      '=stdev(AG5:AG1000)', '=stdev(AH5:AH1000)', '=stdev(AI5:AI1000)', '=stdev(AJ5:AJ1000)',
                      '=stdev(AK5:AK1000)', '=stdev(AL5:AL1000)', '=stdev(AM5:AM1000)', '=stdev(AN5:AN1000)',
                      '=stdev(AO5:AO1000)', '=stdev(AP5:AP1000)', '=stdev(AQ5:AQ1000)', '=stdev(AR5:AR1000)',
                      '=stdev(AS5:AS1000)', '=stdev(AT5:AT1000)', '=stdev(AU5:AU1000)', '=stdev(AV5:AV1000)')
            eval_results_csv.write(header)
            eval_results_csv.flush()

            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_12.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency} config
            care_config_123 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                   SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf)
            care_config_123.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency, actionability} config
            care_config_1234 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True, n_cf=n_cf)
            care_config_1234.fit(X_train, Y_train)

            # explaining instances from test set
            explained = 0
            for x_ord in X_test:

                try:
                    explanation_config_1 = care_config_1.explain(x_ord)
                    explanation_config_12 = care_config_12.explain(x_ord)
                    explanation_config_123 = care_config_123.explain(x_ord)
                    user_preferences = userPreferences(dataset, x_ord)
                    explanation_config_1234 = care_config_1234.explain(x_ord, user_preferences=user_preferences)

                    # evaluating counterfactuals based on all objectives results
                    toolbox = explanation_config_1234['toolbox']
                    objective_names = explanation_config_1234['objective_names']
                    featureScaler = explanation_config_1234['featureScaler']
                    feature_names = dataset['feature_names']

                    # evaluating and recovering counterfactuals of {validity} config
                    cfs_ord_config_1, \
                    cfs_eval_config_1, \
                    x_cfs_ord_config_1, \
                    x_cfs_eval_config_1 = evaluateCounterfactuals(x_ord, explanation_config_1['cfs_ord'], dataset,
                                                                  predict_fn, predict_proba_fn, task, toolbox,
                                                                  objective_names, featureScaler, feature_names)
                    x_org_config_1, \
                    cfs_org_config_1, \
                    x_cfs_org_config_1, \
                    x_cfs_highlight_config_1 = recoverOriginals(x_ord, cfs_ord_config_1, dataset, feature_names)


                    # evaluating and recovering counterfactuals of {validity, soundness} config
                    cfs_ord_config_12, \
                    cfs_eval_config_12, \
                    x_cfs_ord_config_12, \
                    x_cfs_eval_config_12 = evaluateCounterfactuals(x_ord, explanation_config_12['cfs_ord'], dataset,
                                                                   predict_fn, predict_proba_fn, task, toolbox,
                                                                   objective_names, featureScaler, feature_names)
                    x_org_config_12, \
                    cfs_org_config_12, \
                    x_cfs_org_config_12, \
                    x_cfs_highlight_config_12 = recoverOriginals(x_ord, cfs_ord_config_12, dataset, feature_names)


                    # evaluating and recovering counterfactuals of {validity, soundness, coherency} config
                    cfs_ord_config_123, \
                    cfs_eval_config_123, \
                    x_cfs_ord_config_123, \
                    x_cfs_eval_config_123 = evaluateCounterfactuals(x_ord, explanation_config_123['cfs_ord'],
                                                                    dataset, predict_fn, predict_proba_fn, task,
                                                                    toolbox, objective_names, featureScaler,
                                                                    feature_names)
                    x_org_config_123, \
                    cfs_org_config_123, \
                    x_cfs_org_config_123, \
                    x_cfs_highlight_config_123 = recoverOriginals(x_ord, cfs_ord_config_123, dataset, feature_names)


                    # evaluating and recovering counterfactuals of {validity, soundness, coherency, actionability} config
                    cfs_ord_config_1234, \
                    cfs_eval_config_1234, \
                    x_cfs_ord_config_1234, \
                    x_cfs_eval_config_1234 = evaluateCounterfactuals(x_ord, explanation_config_1234['cfs_ord'],
                                                                    dataset, predict_fn, predict_proba_fn, task,
                                                                    toolbox, objective_names, featureScaler,
                                                                    feature_names)
                    x_org_config_1234, \
                    cfs_org_config_1234, \
                    x_cfs_org_config_1234, \
                    x_cfs_highlight_config_1234 = recoverOriginals(x_ord, cfs_ord_config_1234, dataset, feature_names)

                    # finding the index of the best counterfactual in the output of every config
                    idx_best_config_1 = (np.where((x_cfs_ord_config_1 ==
                                               explanation_config_1['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_config_12 = (np.where((x_cfs_ord_config_12 ==
                                                explanation_config_12['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_config_123 = (np.where((x_cfs_ord_config_123 ==
                                                   explanation_config_123['best_cf_ord']).all(axis=1)==True))[0][0]
                    idx_best_config_1234 = (np.where((x_cfs_ord_config_1234 ==
                                                         explanation_config_1234['best_cf_ord']).all(axis=1)==True))[0][0]

                    # storing the best counterfactual found by configs
                    cfs_results = pd.concat([x_cfs_highlight_config_1.iloc[[0,idx_best_config_1]],
                                             x_cfs_eval_config_1.iloc[[0,idx_best_config_1]],
                                             x_cfs_highlight_config_12.iloc[[0,idx_best_config_12]],
                                             x_cfs_eval_config_12.iloc[[0,idx_best_config_12]],
                                             x_cfs_highlight_config_123.iloc[[0,idx_best_config_123]],
                                             x_cfs_eval_config_123.iloc[[0,idx_best_config_123]],
                                             x_cfs_highlight_config_1234.iloc[[0,idx_best_config_1234]],
                                             x_cfs_eval_config_1234.iloc[[0,idx_best_config_1234]]], axis=1)
                    cfs_results.to_csv(cfs_results_csv)
                    cfs_results_csv.write('\n')
                    cfs_results_csv.flush()

                    # storing the evaluation of the best counterfactual found by configs
                    eval_results = np.r_[x_cfs_eval_config_1.iloc[idx_best_config_1, :-n_out],
                                         x_cfs_eval_config_12.iloc[idx_best_config_12, :-n_out],
                                         x_cfs_eval_config_123.iloc[idx_best_config_123, :-n_out],
                                         x_cfs_eval_config_1234.iloc[idx_best_config_1234, :-n_out]]
                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                    explained += 1

                    print('-----------------------------------------------------------------------')
                    print("%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print('-----------------------------------------------------------------------')

                except Exception:
                    pass

                if explained == N:
                    break

            cfs_results_csv.close()
            eval_results_csv.close()

            print('Done!')

if __name__ == '__main__':
    main()
