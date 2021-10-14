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
from dython import nominal
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from care.care import CARE
from user_preferences import userPreferences

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
    }

    # defining the list of black-boxes
    blackbox_list = {
        # 'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    # defining number of samples N and number of counterfactuals generated for every instance n_cf
    experiment_size = {
        'adult': (500, 10),
        'compas-scores-two-years': (500, 10),
        'credit-card-default': (500, 10),
        'heloc': (500,10),
    }

    # defining the features that are used for coherency evaluation
    studied_features ={
        'adult': [0, 1, 6, 7, 8, 9, 10, 11, 12],
        'compas-scores-two-years': [0, 1, 2, 3, 4, 5, 7, 8],
        'credit-card-default': [2, 3, 4, 5, 6, 7, 18, 19, 20, 21, 22],
        'heloc': [0, 4, 5, 6, 15, 16],
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

            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=True, n_cf=n_cf)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=True, n_cf=n_cf)
            care_config_12.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency} config
            care_config_123 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True, n_cf=n_cf)
            care_config_123.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency, actionability} config
            care_config_1234 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True, n_cf=n_cf)
            care_config_1234.fit(X_train, Y_train)

            # generating counterfactuals in which the studied feature will change
            for f in studied_features[dataset_kw]:
                explained = 0
                original_data = []
                care_config_1_cfs = []
                care_config_12_cfs = []
                care_config_123_cfs = []
                care_config_1234_cfs = []
                feature_name = dataset['feature_indices'][f]

                feature_preference = {
                    'constraint': [None for _ in range(X_train.shape[1])],
                    'importance': [None for _ in range(X_train.shape[1])]
                }
                feature_preference['constraint'][f] = 'change'
                feature_preference['importance'][f] = 10

                user_preferences = userPreferences(dataset, X_train[0,:])
                user_preferences['constraint'][f] = 'change'
                user_preferences['importance'][f] = 10

                for x_ord in X_test:
                    # explaining instances
                    explanation_config_1 = care_config_1.explain(x_ord, user_preferences=feature_preference)
                    explanation_config_12 = care_config_12.explain(x_ord, user_preferences=feature_preference)
                    explanation_config_123 = care_config_123.explain(x_ord, user_preferences=feature_preference)
                    explanation_config_1234 = care_config_1234.explain(x_ord, user_preferences=user_preferences)

                    # extracting the best counterfactual
                    care_config_1_best_cf = explanation_config_1['best_cf_ord']
                    care_config_12_best_cf = explanation_config_12['best_cf_ord']
                    care_config_123_best_cf = explanation_config_123['best_cf_ord']
                    care_config_1234_best_cf = explanation_config_1234['best_cf_ord']

                    original_data.append(x_ord)
                    care_config_1_cfs.append(care_config_1_best_cf)
                    care_config_12_cfs.append(care_config_12_best_cf)
                    care_config_123_cfs.append(care_config_123_best_cf)
                    care_config_1234_cfs.append(care_config_1234_best_cf)

                    explained += 1

                    print('----------------------------------------------------------------------------------------')
                    print("%s|%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, feature_name, explained, N))
                    print('----------------------------------------------------------------------------------------')

                    if explained == N:
                        break

                # calculating the coherency preservation rate
                original_data = np.vstack(original_data)
                cf_data = {
                    'care_config_1': np.vstack(care_config_1_cfs),
                    'care_config_12': np.vstack(care_config_12_cfs),
                    'care_config_123': np.vstack(care_config_123_cfs),
                    'care_config_1234': np.vstack(care_config_1234_cfs)
                }

                if os.path.isfile(experiment_path + '%s_%s_original_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name)):
                    os.remove(experiment_path + '%s_%s_original_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name))

                if os.path.isfile(experiment_path + '%s_%s_counterfactual_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name)):
                    os.remove(experiment_path + '%s_%s_counterfactual_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name))

                if os.path.isfile(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name)):
                    os.remove(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name))

                original_data_df = pd.DataFrame(columns=dataset['feature_names'], data=original_data)
                original_corr = nominal.associations(original_data_df, nominal_columns=dataset['discrete_features'])['corr']
                original_corr = original_corr.round(decimals=3)
                original_corr.to_csv(experiment_path + '%s_%s_original_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name))

                for method, cfs in cf_data.items():

                    with open(experiment_path + '%s_%s_counterfactual_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), 'a') as f:
                        f.write(method)
                        f.write('\n')
                    counterfactual_data = np.r_[original_data, cfs]
                    counterfactual_data_df = pd.DataFrame(columns=dataset['feature_names'], data=counterfactual_data)
                    counterfactual_corr = nominal.associations(counterfactual_data_df, nominal_columns=dataset['discrete_features'])['corr']
                    counterfactual_corr = counterfactual_corr.round(decimals=3)
                    counterfactual_corr.to_csv(experiment_path + '%s_%s_counterfactual_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), mode='a')
                    with open(experiment_path + '%s_%s_counterfactual_correlation_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), 'a') as f:
                        f.write('\n')

                    with open(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name),'a') as f:
                        f.write(method)
                        f.write('\n')
                    correlation_diff = np.abs(original_corr - counterfactual_corr)
                    correlation_diff = correlation_diff.round(decimals=3)
                    correlation_diff.to_csv(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), mode='a')

                    with open(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), 'a') as f:
                        f.write('Feature-wise MAE:')
                    correlation_diff.mean().round(3).to_csv(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), mode='a')

                    with open(experiment_path + '%s_%s_correlation_difference_feature_%s.csv' % (dataset_kw, blackbox_name, feature_name), 'a') as f:
                        f.write('Total MAE: ' + str(np.round(correlation_diff.mean().mean(), 3)))
                        f.write('\n \n')

            print('Done!')

if __name__ == '__main__':
    main()
