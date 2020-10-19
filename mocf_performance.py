import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, KerasNeuralNetwork
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
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
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
        # 'boston-house-prices': ('boston-house-prices.csv', PrepareBostonHousePrices, 'regression')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': KerasNeuralNetwork,
        # 'gb-c': GradientBoostingClassifier,
        # 'gb-r': GradientBoostingRegressor
        # 'dt-r': DecisionTreeRegressor,
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

            # classification task
            if task is 'classification':

                # creating an instance of MOCF explainer for  soundCF=False and feasibleAR=False
                explainer_base = MOCF(dataset, task=task, predict_fn=predict_fn,
                                      predict_proba_fn=predict_proba_fn,
                                      soundCF=False, feasibleAR=False, hof_final=True)
                explainer_base.fit(X_train, Y_train)

                # creating an instance of MOCF explainer for  soundCF=True and feasibleAR=False
                explainer_sound = MOCF(dataset, task=task, predict_fn=predict_fn,
                                       predict_proba_fn=predict_proba_fn,
                                       soundCF=True, feasibleAR=False, hof_final=True)
                explainer_sound.fit(X_train, Y_train)

                # creating an instance of MOCF explainer for  soundCF=True and feasibleAR=True
                explainer_sound_feasible = MOCF(dataset, task=task, predict_fn=predict_fn,
                                                predict_proba_fn=predict_proba_fn,
                                                soundCF=True, feasibleAR=True, hof_final=True)
                explainer_sound_feasible.fit(X_train, Y_train)

                ################################### Explaining test samples #########################################
                # setting the size of the experiment
                N = 10  # number of instances to explain
                n_diversity = 5  # number of counter-factuals for measuring diversity

                # creating/opening a csv file for storing results
                exists = os.path.isfile(
                    experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity))
                if exists:
                    os.remove(experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity))
                cfs_results_csv = open(
                    experiment_path + 'mocf_performance_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity), 'a')

                feature_space = ['' for _ in range(X_train.shape[1] - 1 + 9)]
                header = ['', 'Base']
                header += feature_space
                header += ['Sound']
                header += feature_space
                header += ['Sound & Feasible']
                header = ','.join(header)
                cfs_results_csv.write('%s\n' % (header))
                cfs_results_csv.flush()

                # creating/opening a csv file for storing results
                exists = os.path.isfile(
                    experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity))
                if exists:
                    os.remove(experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity))
                eval_results_csv = open(
                    experiment_path + 'mocf_performance_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_diversity), 'a')

                header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                         ('Base', '', '', '', '', '', '', '', '',
                          'Sound', '', '', '', '', '', '', '', '',
                          'Sound & Feasible', '', '', '', '', '', '', '', '')
                eval_results_csv.write(header)

                header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                         ('prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                          'sparsity', 'diversity', 'validity',
                          'prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                          'sparsity', 'diversity', 'validity',
                          'prediction', 'proximity', 'connectedness', 'actionable', 'correlation', 'distance',
                          'sparsity', 'diversity', 'validity')
                eval_results_csv.write(header)

                header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
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
                          '=average(AA5:AA1000)')
                eval_results_csv.write(header)

                header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
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
                          '=stdev(AA5:AA1000)')
                eval_results_csv.write(header)
                eval_results_csv.flush()

                # selecting instances to explain from test set
                np.random.seed(42)
                X_explain = X_test[np.random.choice(range(X_test.shape[0]), size=N, replace=False)]

                # explaining instances
                for x_ord in X_explain:

                    explanation_base = explainer_base.explain(x_ord)
                    explanation_sound = explainer_sound.explain(x_ord)
                    user_preferences = userPreferences(dataset, x_ord)
                    explanation_sound_feasible= explainer_sound_feasible.explain(x_ord, user_preferences=user_preferences)

                    # evaluating counter-factuals based on all objectives results
                    toolbox = explanation_sound_feasible['toolbox']
                    objective_names = explanation_sound_feasible['objective_names']
                    featureScaler = explanation_sound_feasible['featureScaler']
                    feature_names = dataset['feature_names']

                    # evaluating counter-factuals of base method
                    cfs_ord_base, \
                    cfs_eval_base, \
                    x_cfs_ord_base, \
                    x_cfs_eval_base = evaluateCounterfactuals(x_ord, explanation_base['cfs_ord'], dataset,
                                                              predict_fn, predict_proba_fn, task, toolbox,
                                                              objective_names, featureScaler, feature_names)

                    # recovering counter-factuals in original format of base method
                    x_org_base, \
                    cfs_org_base, \
                    x_cfs_org_base, \
                    x_cfs_highlight_base = recoverOriginals(x_ord, cfs_ord_base, dataset, feature_names)


                    # evaluating counter-factuals of sound method
                    cfs_ord_sound, \
                    cfs_eval_sound, \
                    x_cfs_ord_sound, \
                    x_cfs_eval_sound = evaluateCounterfactuals(x_ord, explanation_sound['cfs_ord'], dataset,
                                                               predict_fn, predict_proba_fn, task, toolbox,
                                                               objective_names, featureScaler, feature_names)

                    # recovering counter-factuals in original format of sound method
                    x_org_sound, \
                    cfs_org_sound, \
                    x_cfs_org_sound, \
                    x_cfs_highlight_sound = recoverOriginals(x_ord, cfs_ord_sound, dataset, feature_names)


                    # evaluating counter-factuals of sound and feasible method
                    cfs_ord_sound_feasible, \
                    cfs_eval_sound_feasible, \
                    x_cfs_ord_sound_feasible, \
                    x_cfs_eval_sound_feasible = evaluateCounterfactuals(x_ord, explanation_sound_feasible['cfs_ord'],
                                                                        dataset, predict_fn, predict_proba_fn, task,
                                                                        toolbox, objective_names, featureScaler, feature_names)

                    # recovering counter-factuals in original format of sound and feasible method
                    x_org_sound_feasible, \
                    cfs_org_sound_feasible, \
                    x_cfs_org_sound_feasible, \
                    x_cfs_highlight_sound_feasible = recoverOriginals(x_ord, cfs_ord_sound_feasible, dataset, feature_names)


                    # storing the best counter-factual found by methods
                    cfs_results = pd.concat([x_cfs_highlight_base.iloc[:2], x_cfs_eval_base.iloc[:2],
                                             x_cfs_highlight_sound.iloc[:2], x_cfs_eval_sound.iloc[:2],
                                             x_cfs_highlight_sound_feasible.iloc[:2], x_cfs_eval_sound_feasible.iloc[:2]], axis=1)
                    cfs_results.to_csv(cfs_results_csv)
                    cfs_results_csv.write('\n')
                    cfs_results_csv.flush()

                    # measuring the diversity of counter-factuals using Jaccard metric
                    feature_names_base = []
                    for i in range(n_diversity):
                        feature_names_base.append([dataset['feature_names'][ii] for ii in
                                                   np.where(x_cfs_highlight_base.iloc[i + 1] != '_')[0]])
                    feature_names_base = list(filter(None, feature_names_base))

                    feature_names_sound = []
                    for i in range(n_diversity):
                        feature_names_sound.append([dataset['feature_names'][ii] for ii in
                                                   np.where(x_cfs_highlight_sound.iloc[i + 1] != '_')[0]])
                    feature_names_sound = list(filter(None, feature_names_sound))

                    feature_names_sound_feasible = []
                    for i in range(n_diversity):
                        feature_names_sound_feasible.append([dataset['feature_names'][ii] for ii in
                                                   np.where(x_cfs_highlight_sound_feasible.iloc[i + 1] != '_')[0]])
                    feature_names_sound_feasible = list(filter(None, feature_names_sound_feasible))

                    jaccard_base = []
                    for i in range(0, len(feature_names_base)):
                        for ii in range(i, len(feature_names_base)):
                            jaccard = len(set(feature_names_base[i]) & set(feature_names_base[ii])) / \
                                      len(set(feature_names_base[i]) | set(feature_names_base[ii]))
                            jaccard_base.append(jaccard)

                    jaccard_sound = []
                    for i in range(0, len(feature_names_sound)):
                        for ii in range(i, len(feature_names_sound)):
                            jaccard = len(set(feature_names_sound[i]) & set(feature_names_sound[ii])) / \
                                      len(set(feature_names_sound[i]) | set(feature_names_sound[ii]))
                            jaccard_sound.append(jaccard)

                    jaccard_sound_feasible = []
                    for i in range(0, len(feature_names_sound_feasible)):
                        for ii in range(i, len(feature_names_sound_feasible)):
                            jaccard = len(set(feature_names_sound_feasible[i]) & set(feature_names_sound_feasible[ii])) / \
                                      len(set(feature_names_sound_feasible[i]) | set(feature_names_sound_feasible[ii]))
                            jaccard_sound_feasible.append(jaccard)

                    eval_results = np.r_[cfs_eval_base.iloc[0, :-2], 1.0 - np.mean(jaccard_base), int(cfs_eval_base.iloc[0, 0] == 0),
                                         cfs_eval_sound.iloc[0, :-2], 1.0 - np.mean(jaccard_sound), int(cfs_eval_sound.iloc[0, 0] == 0),
                                         cfs_eval_sound_feasible.iloc[0, :-2], 1.0 - np.mean(jaccard_sound_feasible), int(cfs_eval_sound_feasible.iloc[0, 0] == 0)]


                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                cfs_results_csv.close()
                eval_results_csv.close()

                print('Done!')

            # regression task
            elif task is 'regression':


                print('Done!')

if __name__ == '__main__':
    main()
