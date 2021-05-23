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
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from care.care import CARE
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
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
        'adult': ('adult.csv', PrepareAdult, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
    }

    experiment_size = {
        'adult': (500, 1),
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
                experiment_path + 'benchmark_coherency_preservation_education_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_coherency_preservation_education_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'benchmark_coherency_preservation_education_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = ['CARE',
                      'CFPrototype',
                      'DiCE']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            average = '%s,%s,%s\n' % \
                     ('=average(A3:A1000)', '=average(B3:B1000)', '=average(C3:C1000)')
            eval_results_csv.write(average)
            eval_results_csv.flush()

            # creating explainer instances
            # CARE
            care_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf)
            care_explainer.fit(X_train, Y_train)

            # CFPrototype
            cat_vars_ord = {}
            for i, d in enumerate(dataset['discrete_indices']):
                cat_vars_ord[d] = dataset['n_cat_discrete'][i]
            cat_vars_ohe = ord_to_ohe(X_train, cat_vars_ord)[1]
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
                                                        cat_vars=cat_vars_ohe, ohe=True, beta=0.1, theta=10,
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

            # explaining instances from test set
            # correlation between education-num and education features
            correlations = [(0,13),
                            (1,3),
                            (2,4),
                            (3,5),
                            (4,6),
                            (5,0),
                            (6,1),
                            (7,2),
                            (8,11),
                            (9,15),
                            (10,8),
                            (11,7),
                            (12,9),
                            (13,12),
                            (14,14),
                            (15,10)]
            explained = 0
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE
                    CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=care_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    care_cfs_ord = CARE_output['cfs_ord']
                    education_num = care_cfs_ord['education-num'].to_numpy().astype(int)
                    education = care_cfs_ord['education'].to_numpy().astype(int)
                    preserved_care = 1 if correlations[education_num[0]][1] == education[0] else 0

                    # explain instance x_ord using CFPrototype
                    CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset,
                                                              task, CARE_output, explainer=cfprototype_explainer,
                                                              target_class=None, n_cf=n_cf)
                    cfprototype_cfs_ord = CFPrototype_output['cfs_ord']
                    education_num = cfprototype_cfs_ord['education-num'].to_numpy().astype(int)
                    education = cfprototype_cfs_ord['education'].to_numpy().astype(int)
                    preserved_cfprototype = 1 if correlations[education_num[0]][1] == education[0] else 0

                    # explain instance x_ord using DiCE
                    DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train,
                                                dataset, task, CARE_output, explainer=dice_explainer, ACTIONABILITY=False,
                                                user_preferences=None, n_cf=n_cf, desired_class="opposite",
                                                probability_thresh=0.5, proximity_weight=1.0, diversity_weight=1.0)
                    dice_cfs_ord = DiCE_output['cfs_ord']
                    education_num = dice_cfs_ord['education-num'].to_numpy().astype(int)
                    education = dice_cfs_ord['education'].to_numpy().astype(int)
                    preserved_dice =  1 if correlations[education_num[0]][1] == education[0] else 0


                    print('\n')
                    print('-------------------------------')
                    print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print('\n')
                    print(care_cfs_ord)
                    print(cfprototype_cfs_ord)
                    print(dice_cfs_ord)
                    print('\n')
                    print("preserved coherency | CARE: %0.3f - CFPrototype: %0.3f - DiCE: %0.3f" %
                          (preserved_care, preserved_cfprototype, preserved_dice))
                    print('-----------------------------------------------------------------------')

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[preserved_care, preserved_cfprototype, preserved_dice]
                    eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                    eval_results = ','.join(eval_results)
                    eval_results_csv.write('%s\n' % (eval_results))
                    eval_results_csv.flush()

                    explained += 1

                except Exception:
                    pass

                if explained == N:
                    break

            eval_results_csv.close()

if __name__ == '__main__':
    main()