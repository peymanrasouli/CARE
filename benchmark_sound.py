import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from utils import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, KerasNeuralNetwork
from mocf_explainer import MOCFExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': KerasNeuralNetwork
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

            ################################### Explaining test samples #########################################
            # setting the size of the experiment
            N = 10  # number of instances to explain
            n_cf = 5  # number of counter-factuals for measuring diversity

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_sound_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_sound_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf))
            cfs_results_csv = open(experiment_path + 'benchmark_sound_%s_cfs_%s_%s.csv'%(dataset['name'], N, n_cf), 'a')

            feature_space = ['' for _ in range(X_train.shape[1]-1 + 7)]
            header = ['','MOCF']
            header += feature_space
            header += ['CFPrototype']
            header += feature_space
            header += ['DiCE']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_sound_%s_eval_%s_%s.csv'%(dataset['name'], N, n_cf))
            if exists:
                os.remove(experiment_path + 'benchmark_sound_%s_eval_%s_%s.csv'%(dataset['name'], N, n_cf))
            eval_results_csv = open(experiment_path + 'benchmark_sound_%s_eval_%s_%s.csv'%(dataset['name'], N, n_cf), 'a')

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('MOCF', '', '', '', '', '', '',
                       'CFPrototype', '', '', '', '', '', '',
                       'DiCE', '', '', '', '', '', '')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('prediction', 'proximity', 'connectedness', 'distance', 'sparsity', 'diversity', 'validity',
                       'prediction', 'proximity', 'connectedness', 'distance', 'sparsity', 'diversity', 'validity',
                       'prediction', 'proximity', 'connectedness', 'distance', 'sparsity', 'diversity', 'validity')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
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
                       '=average(U5:U1000)')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
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
                       '=stdev(U5:U1000)')
            eval_results_csv.write(header)
            eval_results_csv.flush()

            # selecting instances to explain from test set
            np.random.seed(42)
            X_explain = X_test[np.random.choice(range(X_test.shape[0]), size=N, replace=False)]

            # explaining instances
            for x_ord in X_explain:

                # explain instance x_ord using MOCF
                MOCF_output = MOCFExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                            soundCF=True, feasibleAR=False, user_preferences=None,
                                            cf_class='opposite', probability_thresh=0.5, n_cf=n_cf)

                mocf_x_cfs_highlight = MOCF_output['x_cfs_highlight']
                mocf_cfs_eval = MOCF_output['cfs_eval']
                mocf_x_cfs_eval = MOCF_output['x_cfs_eval']

                # explain instance x_ord using CFPrototype
                CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset, task,
                                                          MOCF_output, target_class=None, n_cf=n_cf)

                cfprototype_x_cfs_highlight = CFPrototype_output['x_cfs_highlight']
                cfprototype_cfs_eval = CFPrototype_output['cfs_eval']
                cfprototype_x_cfs_eval = CFPrototype_output['x_cfs_eval']

                # explain instance x_ord using DiCE
                DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train, dataset,
                                            task, MOCF_output, feasibleAR=False, user_preferences=None,
                                            n_cf=n_cf, desired_class="opposite", probability_thresh=0.5,
                                            proximity_weight=1.0, diversity_weight=1.0)

                dice_x_cfs_highlight = DiCE_output['x_cfs_highlight']
                dice_cfs_eval = DiCE_output['cfs_eval']
                dice_x_cfs_eval = DiCE_output['x_cfs_eval']


                # storing the best counter-factual found by methods
                cfs_results = pd.concat([mocf_x_cfs_highlight.iloc[:2], mocf_x_cfs_eval.iloc[:2],
                                         cfprototype_x_cfs_highlight.iloc[:2], cfprototype_x_cfs_eval.iloc[:2],
                                         dice_x_cfs_highlight.iloc[:2], dice_x_cfs_eval.iloc[:2]], axis=1)
                cfs_results.to_csv(cfs_results_csv)
                cfs_results_csv.write('\n')
                cfs_results_csv.flush()

                # measuring the diversity of counter-factuals using Jaccard metric
                n_cf_mocf = min(n_cf, mocf_cfs_eval.shape[0])
                n_cf_cfprototype = min(n_cf, cfprototype_cfs_eval.shape[0])
                n_cf_dice= min(n_cf, dice_cfs_eval.shape[0])

                mocf_feature_names = []
                for i in range(n_cf_mocf):
                    mocf_feature_names.append([dataset['feature_names'][ii] for ii in np.where(mocf_x_cfs_highlight.iloc[i+1] != '_')[0]])
                mocf_feature_names = list(filter(None, mocf_feature_names))

                cfprototype_feature_names = []
                for i in range(n_cf_cfprototype):
                    cfprototype_feature_names.append([dataset['feature_names'][ii] for ii in np.where(cfprototype_x_cfs_highlight.iloc[i + 1] != '_')[0]])
                cfprototype_feature_names = list(filter(None, cfprototype_feature_names))

                dice_feature_names = []
                for i in range(n_cf_dice):
                    dice_feature_names.append([dataset['feature_names'][ii] for ii in np.where(dice_x_cfs_highlight.iloc[i + 1] != '_')[0]])
                dice_feature_names = list(filter(None, dice_feature_names))

                mocf_jaccard = []
                for i in range(0, len(mocf_feature_names)):
                    for ii in range(i, len(mocf_feature_names)):
                        jaccard = len(set(mocf_feature_names[i]) & set(mocf_feature_names[ii])) / \
                                  len(set(mocf_feature_names[i]) | set(mocf_feature_names[ii]))
                        mocf_jaccard.append(jaccard)

                cfprototype_jaccard = []
                for i in range(0, len(cfprototype_feature_names)):
                    for ii in range(i, len(cfprototype_feature_names)):
                        jaccard = len(set(cfprototype_feature_names[i]) & set(cfprototype_feature_names[ii])) / \
                                  len(set(cfprototype_feature_names[i]) | set(cfprototype_feature_names[ii]))
                        cfprototype_jaccard.append(jaccard)

                dice_jaccard = []
                for i in range(0, len(dice_feature_names)):
                    for ii in range(i, len(dice_feature_names)):
                        jaccard = len(set(dice_feature_names[i]) & set(dice_feature_names[ii])) / \
                                  len(set(dice_feature_names[i]) | set(dice_feature_names[ii]))
                        dice_jaccard.append(jaccard)

                eval_results = np.r_[mocf_cfs_eval.iloc[0, :-2], 1.0 - np.mean(mocf_jaccard), int(mocf_cfs_eval.iloc[0, 0] == 0),
                                     cfprototype_cfs_eval.iloc[0, :-2], 1.0 - np.mean(cfprototype_jaccard), int(cfprototype_cfs_eval.iloc[0, 0] == 0),
                                     dice_cfs_eval.iloc[0, :-2], 1.0 - np.mean(dice_jaccard), int(dice_cfs_eval.iloc[0, 0] == 0)]

                eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                eval_results = ','.join(eval_results)
                eval_results_csv.write('%s\n' % (eval_results))
                eval_results_csv.flush()

            cfs_results_csv.close()
            eval_results_csv.close()

if __name__ == '__main__':
    main()
