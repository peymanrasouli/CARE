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
from care.care import CARE
from alibi.explainers import CounterFactualProto
import dice_ml
from care_explainer import CAREExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer

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
    }

    experiment_size = {
        'simple-binomial': (200, 1)
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
            predict_fn = lambda x: blackbox.predict_classes(x).ravel()
            predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_causality_preservation_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_causality_preservation_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf))
            cfs_results_csv = open(experiment_path + 'benchmark_causality_preservation_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf), 'a')

            n_out = int(task == 'classification') + 1
            n_metrics = 11
            feature_space = ['' for _ in range(X_train.shape[1] - 1 + n_metrics + n_out)]
            header = ['','CARE']
            header += feature_space
            header += ['CFPrototype']
            header += feature_space
            header += ['DiCE']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'benchmark_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf))
            eval_results_csv = open(
                experiment_path + 'benchmark_causality_preservation_%s_eval_%s_%s.csv' % (dataset['name'], N, n_cf), 'a')

            header = ['CARE', '', '',
                      'CFPrototype', '', '',
                      'DiCE', '', '',]
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))

            header = ['n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate',
                      'n_causes', 'n_effect', 'preservation_rate']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            eval_results_csv.flush()

            # creating explainer instances
            # CARE
            care_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  sound=True, causality=True, actionable=False, n_cf=n_cf,
                                  corr_thresh=0.0001, corr_model_score_thresh=0.7)
            care_explainer.fit(X_train, Y_train)

            # CFPrototype
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
                                                        ohe=False, beta=0.1, theta=10,
                                                        use_kdtree=True, max_iterations=500, c_init=1.0, c_steps=5)
            X_train_ohe = ord2ohe(X_train, dataset)
            cfprototype_explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

            # DiCE
            feature_names = dataset['feature_names']
            continuous_features = dataset['continuous_features']
            data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names + ['class'])
            data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
            data = dice_ml.Data(dataframe=data_frame,
                                continuous_features=continuous_features,
                                outcome_name='class')
            backend = 'TF1'
            model = dice_ml.Model(model=blackbox, backend=backend)
            dice_explainer = dice_ml.Dice(data, model)

            # explaining instances from test set
            explained = 0
            care_preservation = 0.0
            cfprototype_preservation = 0.0
            dice_preservation = 0.0
            care_n_causes = []
            cfprototype_n_causes = []
            dice_n_causes = []
            care_n_effects = []
            cfprototype_n_effects = []
            dice_n_effects = []
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE
                    CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=care_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    care_x_cfs_highlight = CARE_output['x_cfs_highlight']
                    care_x_cfs_eval = CARE_output['x_cfs_eval']
                    care_cfs_ord = CARE_output['cfs_ord']

                    # explain instance x_ord using CFPrototype
                    CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset,
                                                              task, CARE_output, explainer=cfprototype_explainer,
                                                              target_class=None, n_cf=n_cf)
                    cfprototype_x_cfs_highlight = CFPrototype_output['x_cfs_highlight']
                    cfprototype_x_cfs_eval = CFPrototype_output['x_cfs_eval']
                    cfprototype_cfs_ord = CFPrototype_output['cfs_ord']

                    # explain instance x_ord using DiCE
                    DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train,
                                                dataset, task, CARE_output, explainer=dice_explainer, actionable=False,
                                                user_preferences=None, n_cf=n_cf, desired_class="opposite",
                                                probability_thresh=0.5, proximity_weight=1.0, diversity_weight=1.0)
                    dice_x_cfs_highlight = DiCE_output['x_cfs_highlight']
                    dice_x_cfs_eval = DiCE_output['x_cfs_eval']
                    dice_cfs_ord = DiCE_output['cfs_ord']

                    # print counterfactuals and their corresponding objective values
                    print('\n')
                    print('CARE Results')
                    print(pd.concat([care_x_cfs_highlight, care_x_cfs_eval], axis=1))
                    print('\n')
                    print('CFProto Results')
                    print(pd.concat([cfprototype_x_cfs_highlight, cfprototype_x_cfs_eval], axis=1))
                    print('\n')
                    print('DiCE Results')
                    print(pd.concat([dice_x_cfs_highlight, dice_x_cfs_eval], axis=1))
                    print('\n')


                    # N.B. third features determined by a monotonically increasing/decreasing function of
                    # first and second features, therefore,
                    # (1st feature and 2nd feature) increase => 3rd feature increases
                    # (1st feature and 2nd feature) decrease => 3rd feature decreases

                    # calculating the number of counterfactuals that preserved causality in CARE method
                    care_causes = np.logical_or(
                        np.logical_and(care_cfs_ord.iloc[:, 0] > x_ord[0], care_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(care_cfs_ord.iloc[:, 0] < x_ord[0], care_cfs_ord.iloc[:, 1] < x_ord[1]))
                    care_effects = np.logical_or(np.logical_and(np.logical_and(care_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          care_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           care_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(care_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          care_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           care_cfs_ord.iloc[:, 2] < x_ord[2]))
                    if sum(care_causes) == 0:
                        pass
                    else:
                        care_n_causes.append(sum(care_causes))
                        care_n_effects.append(sum(care_effects))
                        care_preservation = sum(care_n_effects) / sum(care_n_causes)

                    # calculating the number of counterfactuals that preserved causality in sound and CFPrototype method
                    cfprototype_causes = np.logical_or(np.logical_and(cfprototype_cfs_ord.iloc[:, 0] > x_ord[0],
                                                          cfprototype_cfs_ord.iloc[:, 1] > x_ord[1]),
                                           np.logical_and(cfprototype_cfs_ord.iloc[:, 0] < x_ord[0],
                                                          cfprototype_cfs_ord.iloc[:, 1] < x_ord[1]))
                    cfprototype_effects = np.logical_or(np.logical_and(np.logical_and(cfprototype_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          cfprototype_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           cfprototype_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(cfprototype_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          cfprototype_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           cfprototype_cfs_ord.iloc[:, 2] < x_ord[2]))

                    if sum(cfprototype_causes) == 0:
                        pass
                    else:
                        cfprototype_n_causes.append(sum(cfprototype_causes))
                        cfprototype_n_effects.append(sum(cfprototype_effects))
                        cfprototype_preservation = sum(cfprototype_n_effects) / sum(cfprototype_n_causes)


                    # calculating the number of counterfactuals that preserved causality in DiCE method
                    dice_causes = np.logical_or(
                        np.logical_and(dice_cfs_ord.iloc[:, 0] > x_ord[0], dice_cfs_ord.iloc[:, 1] > x_ord[1]),
                        np.logical_and(dice_cfs_ord.iloc[:, 0] < x_ord[0], dice_cfs_ord.iloc[:, 1] < x_ord[1]))
                    dice_effects = np.logical_or(np.logical_and(np.logical_and(dice_cfs_ord.iloc[:, 0] > x_ord[0],
                                                                          dice_cfs_ord.iloc[:, 1] > x_ord[1]),
                                                           dice_cfs_ord.iloc[:, 2] > x_ord[2]),
                                            np.logical_and(np.logical_and(dice_cfs_ord.iloc[:, 0] < x_ord[0],
                                                                          dice_cfs_ord.iloc[:, 1] < x_ord[1]),
                                                           dice_cfs_ord.iloc[:, 2] < x_ord[2]))

                    if sum(dice_causes) == 0:
                        pass
                    else:
                        dice_n_causes.append(sum(dice_causes))
                        dice_n_effects.append(sum(dice_effects))
                        dice_preservation = sum(dice_n_effects) / sum(dice_n_causes)

                    explained += 1

                    print('\n')
                    print('-----------------------------------------------------------------------')
                    print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print("preserved causality | CARE: %0.3f - CFPrototype: %0.3f - DiCE: %0.3f" %
                          (care_preservation, cfprototype_preservation, dice_preservation))
                    print('-----------------------------------------------------------------------')

                    # storing the best counterfactual found by methods
                    cfs_results = pd.concat([care_x_cfs_highlight.iloc[:2], care_x_cfs_eval.iloc[:2],
                                             cfprototype_x_cfs_highlight.iloc[:2], cfprototype_x_cfs_eval.iloc[:2],
                                             dice_x_cfs_highlight.iloc[:2], dice_x_cfs_eval.iloc[:2]], axis=1)
                    cfs_results.to_csv(cfs_results_csv)
                    cfs_results_csv.write('\n')
                    cfs_results_csv.flush()

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[np.sum(care_n_causes), np.sum(care_n_effects), care_preservation,
                                         np.sum(cfprototype_n_causes), np.sum(cfprototype_n_effects), cfprototype_preservation,
                                         np.sum(dice_n_causes), np.sum(dice_n_effects), dice_preservation]
                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                except Exception:
                    pass

                if explained == N:
                    break

            cfs_results_csv.close()
            eval_results_csv.close()

if __name__ == '__main__':
    main()