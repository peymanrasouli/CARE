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
        'simple-binomial': ('simple-binomial', PrepareSimpleBinomial, 'classification')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    experiment_size = {
        'simple-binomial': (500, 10)
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

            # creating an instance of CARE explainer with sound=True
            sound_explainer = CARE(dataset, task=task, predict_fn=predict_fn,
                                   predict_proba_fn=predict_proba_fn,
                                   sound=True, causality=False, actionable=False,
                                   corr_model_score_thresh=0.2, n_cf=n_cf)

            # creating an instance of CARE explainer with sound=True and causality=True
            sound_causality_explainer = CARE(dataset, task=task, predict_fn=predict_fn,
                                             predict_proba_fn=predict_proba_fn,
                                             sound=True, causality=True, actionable=False,
                                             corr_model_score_thresh=0.2, n_cf=n_cf)

            # fitting the sound explainer on the training data
            sound_explainer.fit(X_train, Y_train)

            # fitting the sound and causality explainer on the training data
            sound_causality_explainer.fit(X_train, Y_train)

            # explaining test samples
            explained = 0
            sound_preservation = []
            sound_causality_preservation = []
            for x_ord in X_test:

                # generating counterfactuals
                explanation_sound = sound_explainer.explain(x_ord)
                explanation_sound_causality = sound_causality_explainer.explain(x_ord)

                # extracting objects for evaluation
                toolbox = explanation_sound_causality['toolbox']
                objective_names = explanation_sound_causality['objective_names']
                featureScaler = explanation_sound_causality['featureScaler']
                feature_names = dataset['feature_names']

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

                # evaluating and recovering counterfactuals of sound and causality method
                cfs_ord_sound_causality, \
                cfs_eval_sound_causality, \
                x_cfs_ord_sound_causality, \
                x_cfs_eval_sound_causality = evaluateCounterfactuals(x_ord, explanation_sound_causality['cfs_ord'],
                                                                     dataset, predict_fn, predict_proba_fn, task,
                                                                     toolbox, objective_names, featureScaler,
                                                                     feature_names)
                x_org_sound_causality, \
                cfs_org_sound_causality, \
                x_cfs_org_sound_causality, \
                x_cfs_highlight_sound_causality = recoverOriginals(x_ord, cfs_ord_sound_causality, dataset,
                                                                   feature_names)


                # print counterfactuals and their corresponding objective values
                print('\n')
                print('Sound Results')
                print(pd.concat([x_cfs_highlight_sound, x_cfs_eval_sound], axis=1))
                print('\n')
                print('Sound & Causality Results')
                print(pd.concat([x_cfs_highlight_sound_causality, x_cfs_eval_sound_causality], axis=1))
                print('\n')

                # storing the counterfactuals
                action_series_results = pd.concat([x_cfs_highlight_sound, x_cfs_highlight_sound_causality,
                                                   x_cfs_eval_sound, x_cfs_eval_sound_causality], axis=1)
                action_series_results.to_csv(action_series_results_csv)
                action_series_results_csv.write('\n')
                action_series_results_csv.flush()

                # N.B. third features determined by a monotonically increasing/decreasing function of
                # first and second features, therefore,
                # (1st feature and 2nd feature) increase => 3rd feature increases
                # (1st feature and 2nd feature) decrease => 3rd feature decreases

                # calculating the number of counterfactuals that preserved causality in sound method
                causes = np.logical_or(
                    np.logical_and(cfs_ord_sound.iloc[:, 0] > x_ord[0], cfs_ord_sound.iloc[:, 1] > x_ord[1]),
                    np.logical_and(cfs_ord_sound.iloc[:, 0] < x_ord[0], cfs_ord_sound.iloc[:, 1] < x_ord[1]))
                effects = np.logical_or(np.logical_and(np.logical_and(cfs_ord_sound.iloc[:, 0] > x_ord[0],
                                                                      cfs_ord_sound.iloc[:, 1] > x_ord[1]),
                                                       cfs_ord_sound.iloc[:, 2] > x_ord[2]),
                                        np.logical_and(np.logical_and(cfs_ord_sound.iloc[:, 0] < x_ord[0],
                                                                      cfs_ord_sound.iloc[:, 1] < x_ord[1]),
                                                       cfs_ord_sound.iloc[:, 2] < x_ord[2]))
                if sum(causes) == 0:
                    pass
                else:
                    sound_preservation.append(sum(effects) / sum(causes))

                # calculating the number of counterfactuals that preserved causality in sound and causality method
                causes = np.logical_or(np.logical_and(cfs_ord_sound_causality.iloc[:, 0] > x_ord[0],
                                                      cfs_ord_sound_causality.iloc[:, 1] > x_ord[1]),
                                       np.logical_and(cfs_ord_sound_causality.iloc[:, 0] < x_ord[0],
                                                      cfs_ord_sound_causality.iloc[:, 1] < x_ord[1]))
                effects = np.logical_or(np.logical_and(np.logical_and(cfs_ord_sound_causality.iloc[:, 0] > x_ord[0],
                                                                      cfs_ord_sound_causality.iloc[:, 1] > x_ord[1]),
                                                       cfs_ord_sound_causality.iloc[:, 2] > x_ord[2]),
                                        np.logical_and(np.logical_and(cfs_ord_sound_causality.iloc[:, 0] < x_ord[0],
                                                                      cfs_ord_sound_causality.iloc[:, 1] < x_ord[1]),
                                                       cfs_ord_sound_causality.iloc[:, 2] < x_ord[2]))
                if sum(causes) == 0:
                    pass
                else:
                    sound_causality_preservation.append(sum(effects) / sum(causes))

                explained += 1

                print('\n')
                print('-----------------------------------------------------------------------')
                print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                print("preserved causality | sound: %0.3f - sound_causality: %0.3f" %
                      (np.mean(sound_preservation), np.mean(sound_causality_preservation)))
                print('-----------------------------------------------------------------------')

                if explained == N:
                    break

            print('\n')
            print('Done!')

if __name__ == '__main__':
    main()
