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

    # defining number of samples N and number of counterfactuals generated for every instance n_cf
    experiment_size = {
        'adult': (500, 10),
        'compas-scores-two-years': (500, 10),
        'credit-card-default': (500, 10),
        'heloc': (500,10),
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
            exists = os.path.isfile(experiment_path + 'benchmark_validity_soundness_%s_cfs_%s_%s.csv'% (dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_validity_soundness_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf))
            cfs_results_csv = open(experiment_path + 'benchmark_validity_soundness_%s_cfs_%s_%s.csv'% (dataset['name'], N, n_cf), 'a')

            n_out = int(task == 'classification') + 1
            n_metrics = 10
            feature_space = ['' for _ in range(X_train.shape[1] - 1 + n_metrics + n_out)]
            header = ['','CARE']
            header += feature_space
            header += ['CFPrototype']
            header += feature_space
            header += ['DiCE']
            header += feature_space
            header += ['CERTIFAI']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_validity_soundness_%s_eval_%s_%s.csv'% (dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_validity_soundness_%s_eval_%s_%s.csv'%(dataset['name'], N, n_cf))
            eval_results_csv = open(experiment_path + 'benchmark_validity_soundness_%s_eval_%s_%s.csv'% (dataset['name'], N, n_cf), 'a')

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('CARE', '', '', '', '', '', '', '', '', '',
                       'CFPrototype', '', '', '', '', '', '', '', '', '',
                       'DiCE', '', '', '', '', '', '', '', '', '',
                       'CERTIFAI', '', '', '', '', '', '', '', '', '')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('Outcome', 'Proximity', 'Connectedness', 'Distance', 'Sparsity', 'i-Validity', 's-Validity',
                       'f-Diversity', 'v-Diversity', 'd-Diversity',
                       'Outcome', 'Proximity', 'Connectedness', 'Distance', 'Sparsity', 'i-Validity', 's-Validity',
                       'f-Diversity', 'v-Diversity', 'd-Diversity',
                       'Outcome', 'Proximity', 'Connectedness', 'Distance', 'Sparsity', 'i-Validity', 's-Validity',
                       'f-Diversity', 'v-Diversity', 'd-Diversity',
                       'Outcome', 'Proximity', 'Connectedness', 'Distance', 'Sparsity', 'i-Validity', 's-Validity',
                       'f-Diversity', 'v-Diversity', 'd-Diversity')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=average(A5:A1000)', '=average(B5:B1000)', '=average(C5:C1000)', '=average(D5:D1000)',
                      '=average(E5:E1000)', '=average(F5:F1000)', '=average(G5:G1000)', '=average(H5:H1000)',
                      '=average(I5:I1000)', '=average(J5:J1000)', '=average(K5:K1000)', '=average(L5:L1000)',
                      '=average(M5:M1000)', '=average(N5:N1000)', '=average(O5:O1000)', '=average(P5:P1000)',
                      '=average(Q5:Q1000)', '=average(R5:R1000)', '=average(S5:S1000)', '=average(T5:T1000)',
                      '=average(U5:U1000)', '=average(V5:V1000)', '=average(W5:W1000)', '=average(X5:X1000)',
                      '=average(Y5:Y1000)', '=average(Z5:Z1000)', '=average(AA5:AA1000)','=average(AB5:AB1000)',
                      '=average(AC5:AC1000)','=average(AD5:AD1000)', '=average(AE5:AE1000)', '=average(AF5:AF1000)',
                      '=average(AG5:AG1000)', '=average(AH5:AH1000)', '=average(AI5:AI1000)', '=average(AJ5:AJ1000)',
                      '=average(AK5:AK1000)', '=average(AL5:AL1000)', '=average(AM5:AM1000)', '=average(AN5:AN1000)')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                     '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=stdev(A5:A1000)', '=stdev(B5:B1000)', '=stdev(C5:C1000)', '=stdev(D5:D1000)',
                      '=stdev(E5:E1000)', '=stdev(F5:F1000)', '=stdev(G5:G1000)', '=stdev(H5:H1000)',
                      '=stdev(I5:I1000)', '=stdev(J5:J1000)', '=stdev(K5:K1000)', '=stdev(L5:L1000)',
                      '=stdev(M5:M1000)', '=stdev(N5:N1000)', '=stdev(O5:O1000)', '=stdev(P5:P1000)',
                      '=stdev(Q5:Q1000)', '=stdev(R5:R1000)', '=stdev(S5:S1000)', '=stdev(T5:T1000)',
                      '=stdev(U5:U1000)', '=stdev(V5:V1000)', '=stdev(W5:W1000)', '=stdev(X5:X1000)',
                      '=stdev(Y5:Y1000)', '=stdev(Z5:Z1000)', '=stdev(AA5:AA1000)','=stdev(AB5:AB1000)',
                      '=stdev(AC5:AC1000)','=stdev(AD5:AD1000)', '=stdev(AE5:AE1000)', '=stdev(AF5:AF1000)',
                      '=stdev(AG5:AG1000)', '=stdev(AH5:AH1000)', '=stdev(AI5:AI1000)', '=stdev(AJ5:AJ1000)',
                      '=stdev(AK5:AK1000)', '=stdev(AL5:AL1000)', '=stdev(AM5:AM1000)', '=stdev(AN5:AN1000)'
                      )
            eval_results_csv.write(header)
            eval_results_csv.flush()

            # creating explainer instances
            # CARE with {validity, soundness} config
            care_explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
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

            # explaining instances from test set
            explained = 0
            for x_ord in X_test:

                try:
                    # explain instance x_ord using CARE
                    CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                predict_proba_fn, explainer=care_explainer,
                                                cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)
                    care_x_cfs_highlight = CARE_output['x_cfs_highlight']
                    care_cfs_eval = CARE_output['cfs_eval']
                    care_x_cfs_eval = CARE_output['x_cfs_eval']

                    # explain instance x_ord using CFPrototype
                    CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset,
                                                              task, CARE_output, explainer=cfprototype_explainer,
                                                              target_class=None, n_cf=n_cf)
                    cfprototype_x_cfs_highlight = CFPrototype_output['x_cfs_highlight']
                    cfprototype_cfs_eval = CFPrototype_output['cfs_eval']
                    cfprototype_x_cfs_eval = CFPrototype_output['x_cfs_eval']

                    # explain instance x_ord using DiCE
                    DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train,
                                                dataset, task, CARE_output, explainer=dice_explainer, ACTIONABILITY=False,
                                                user_preferences=None, n_cf=n_cf, desired_class="opposite",
                                                probability_thresh=0.5, proximity_weight=1.0, diversity_weight=1.0)
                    dice_x_cfs_highlight = DiCE_output['x_cfs_highlight']
                    dice_cfs_eval = DiCE_output['cfs_eval']
                    dice_x_cfs_eval = DiCE_output['x_cfs_eval']

                    # explain instance x_ord using CERTIFAI
                    CERTIFAI_output = CERTIFAIExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                        predict_proba_fn, CARE_output, explainer=certifai_explainer,
                                                        cf_class='opposite', n_cf=n_cf)
                    certifai_x_cfs_highlight = CERTIFAI_output['x_cfs_highlight']
                    certifai_cfs_eval = CERTIFAI_output['cfs_eval']
                    certifai_x_cfs_eval = CERTIFAI_output['x_cfs_eval']

                    # storing the best counterfactual found by methods
                    cfs_results = pd.concat([care_x_cfs_highlight.iloc[:2], care_x_cfs_eval.iloc[:2],
                                             cfprototype_x_cfs_highlight.iloc[:2], cfprototype_x_cfs_eval.iloc[:2],
                                             dice_x_cfs_highlight.iloc[:2], dice_x_cfs_eval.iloc[:2],
                                             certifai_x_cfs_highlight.iloc[:2], certifai_x_cfs_eval.iloc[:2]], axis=1)
                    cfs_results.to_csv(cfs_results_csv)
                    cfs_results_csv.write('\n')
                    cfs_results_csv.flush()

                    # storing the evaluation of the best counterfactual found by methods
                    eval_results = np.r_[care_cfs_eval.iloc[0, :-n_out],
                                         cfprototype_cfs_eval.iloc[0, :-n_out],
                                         dice_cfs_eval.iloc[0, :-n_out],
                                         certifai_cfs_eval.iloc[0, :-n_out]]
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

if __name__ == '__main__':
    main()