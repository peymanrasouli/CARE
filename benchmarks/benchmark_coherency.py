import os
import sys
sys.path.append("../")
sys.path.insert(0, "../alibi")
sys.path.insert(0, "../DiCE")
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
from care.care import CARE
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
import dice_ml
from certifai.certifai import CERTIFAI
from care_explainer import CAREExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer
from certifai_explainer import CERTIFAIExplainer

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
        'nn-c': MLPClassifier,
    }

    experiment_size = {
        'adult': (500, 1),
        'compas-scores-two-years': (500, 1),
        'credit-card-default': (500, 1),
        'heloc': (500,1),
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

            # creating explainer instances
            # CARE
            care_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_population=200, n_cf=n_cf)
            care_explainer.fit(X_train, Y_train)

            # CFPrototype
            cat_vars_ord = {}
            for i, d in enumerate(dataset['discrete_indices']):
                cat_vars_ord[d] = dataset['n_cat_discrete'][i]
            cat_vars_ohe = ord_to_ohe(X_train, cat_vars_ord)[1]
            ohe = True if dataset['discrete_availability'] else False
            x_ohe = ord2ohe(X_train[0], dataset)
            x_ohe = x_ohe.reshape((1,) + x_ohe.shape)
            shape = x_ohe.shape
            rng_min = np.min(X_train, axis=0)
            rng_max = np.max(X_train, axis=0)
            rng = tuple([rng_min.reshape(1, -1), rng_max.reshape(1, -1)])
            rng_shape = (1,) + X_train.shape[1:]
            feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32),
                             (np.ones(rng_shape) * rng[1]).astype(np.float32))
            cfprototype_explainer = CounterFactualProto(predict=predict_proba_fn, shape=shape,
                                                        feature_range=feature_range,
                                                        cat_vars=cat_vars_ohe, ohe=ohe, beta=0.1, theta=10,
                                                        use_kdtree=True, max_iterations=500, c_init=1.0, c_steps=5)
            X_train_ohe = ord2ohe(X_train, dataset)
            cfprototype_explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

            # DiCE
            feature_names = dataset['feature_names']
            continuous_features = dataset['continuous_features']
            discrete_features = dataset['discrete_features']
            data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names + ['class'])
            data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
            data_frame[discrete_features] = (data_frame[discrete_features]).astype(int)
            data = dice_ml.Data(dataframe=data_frame,
                                continuous_features=continuous_features,
                                outcome_name='class')
            backend = 'TF1'
            model = dice_ml.Model(model=blackbox, backend=backend)
            dice_explainer = dice_ml.Dice(data, model)

            # CERTIFAI
            certifai_explainer = CERTIFAI(dataset, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                          ACTIONABILITY=False, n_population=100, n_generation=50, n_cf=n_cf)
            certifai_explainer.fit(X_train, Y_train)

            explained = 0
            original_data = []
            care_cfs = []
            cfprototype_cfs = []
            dice_cfs = []
            certifai_cfs = []
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE
                    CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=care_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    care_best_cf = CARE_output['best_cf_ord']
                    if int(CARE_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CARE_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using CFPrototype
                    CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset,
                                                              task, CARE_output, explainer=cfprototype_explainer,
                                                              target_class=None, n_cf=n_cf)
                    cfprototype_best_cf = CFPrototype_output['best_cf_ord']
                    if int(CFPrototype_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CFPrototype_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using DiCE
                    DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train,
                                                dataset, task, CARE_output, explainer=dice_explainer, ACTIONABILITY=False,
                                                user_preferences=None, n_cf=n_cf, desired_class="opposite",
                                                probability_thresh=0.5, proximity_weight=1.0, diversity_weight=1.0)
                    dice_best_cf = DiCE_output['best_cf_ord']
                    if int(DiCE_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(DiCE_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using CERTIFAI
                    CERTIFAI_output = CERTIFAIExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                         predict_proba_fn, CARE_output, explainer=certifai_explainer,
                                                         cf_class='opposite', n_cf=n_cf)
                    certifai_best_cf = CERTIFAI_output['best_cf_ord']
                    if int(CERTIFAI_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CERTIFAI_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    original_data.append(x_ord)
                    care_cfs.append(care_best_cf)
                    cfprototype_cfs.append(cfprototype_best_cf)
                    dice_cfs.append(dice_best_cf)
                    certifai_cfs.append(certifai_best_cf)

                    explained += 1

                    print('-----------------------------------------------------------------------')
                    print("%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print('-----------------------------------------------------------------------')

                except Exception:
                    pass

                if explained == N:
                    break

            # calculating the coherency presenvation rate
            original_data = np.vstack(original_data)
            cf_data = {
                'care': np.vstack(care_cfs),
                'cfprototype': np.vstack(cfprototype_cfs),
                'dice': np.vstack(dice_cfs),
                'certifai': np.vstack(certifai_cfs)
                }

            if os.path.isfile(experiment_path + '%s_%s_original_correlation.csv' % (dataset_kw, blackbox_name)):
                os.remove(experiment_path + '%s_%s_original_correlation.csv' % (dataset_kw, blackbox_name))

            if os.path.isfile(experiment_path + '%s_%s_counterfactual_correlation.csv' % (dataset_kw, blackbox_name)):
                os.remove(experiment_path + '%s_%s_counterfactual_correlation.csv' % (dataset_kw, blackbox_name))

            if os.path.isfile(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name)):
                os.remove(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name))

            original_data_df = pd.DataFrame(columns=dataset['feature_names'], data=original_data)
            original_corr = nominal.associations(original_data_df, nominal_columns=dataset['discrete_features'])['corr']
            original_corr = original_corr.round(decimals=3)
            original_corr.to_csv(experiment_path + '%s_%s_original_correlation.csv' % (dataset_kw, blackbox_name))

            for method, cfs in cf_data.items():

                with open(experiment_path + '%s_%s_counterfactual_correlation.csv' % (dataset_kw, blackbox_name), 'a') as f:
                    f.write(method)
                    f.write('\n')
                counterfactual_data = np.r_[original_data, cfs]
                counterfactual_data_df = pd.DataFrame(columns=dataset['feature_names'], data=counterfactual_data)
                counterfactual_corr = nominal.associations(counterfactual_data_df, nominal_columns=dataset['discrete_features'])['corr']
                counterfactual_corr = counterfactual_corr.round(decimals=3)
                counterfactual_corr.to_csv(experiment_path + '%s_%s_counterfactual_correlation.csv' % (dataset_kw, blackbox_name), mode='a')
                with open(experiment_path + '%s_%s_counterfactual_correlation.csv' % (dataset_kw, blackbox_name), 'a') as f:
                    f.write('\n')

                with open(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name), 'a') as f:
                    f.write(method)
                    f.write('\n')
                correlation_diff = np.abs(original_corr-counterfactual_corr)
                correlation_diff = correlation_diff.round(decimals=3)
                correlation_diff.to_csv(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name), mode='a')

                with open(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name), 'a') as f:
                    f.write('Feature-wise MAE:')
                correlation_diff.mean().round(3).to_csv(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name), mode='a')

                with open(experiment_path + '%s_%s_correlation_difference.csv' % (dataset_kw, blackbox_name), 'a') as f:
                    f.write('Total MAE: ' + str(np.round(correlation_diff.mean().mean(),3)))
                    f.write('\n \n')

            print('Done!')

if __name__ == '__main__':
    main()