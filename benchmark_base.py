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
from user_preferences import userPreferences
from mocf import MOCF
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
import dice_ml
import tensorflow as tf
tf.InteractiveSession()
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
    }

    # defining the list of black-boxes
    blackbox_list = {
        'dnn': KerasNeuralNetwork
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

            ################################### Creating explainers #########################################
            ## MOCF explainer
            # creating an instance of MOCF explainer
            mocf_explainer = MOCF(dataset, task=task, predict_fn=predict_fn,
                                  predict_proba_fn=predict_proba_fn,
                                  soundCF=False, feasibleAR=False, hof_final=False)

            # fitting the explainer on the training data
            mocf_explainer.fit(X_train, Y_train)

            ## CFPrototype explainer
            # preparing parameters
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

            # creating prototype counter-factual explainer
            cfprototype_explainer = CounterFactualProto(predict=predict_proba_fn, shape=shape,
                                                         feature_range=feature_range,
                                                         cat_vars=cat_vars_ohe, ohe=True)

            # Fitting the explainer on the training data
            X_train_ohe = ord2ohe(X_train, dataset)
            cfprototype_explainer.fit(X_train_ohe, d_type='abdm', disc_perc=[25, 50, 75])

            ## DiCE explainer
            # preparing ataset for DiCE model
            feature_names = dataset['feature_names']
            continuous_features = dataset['continuous_features']
            discrete_features = dataset['discrete_features']

            data_frame = pd.DataFrame(data=np.c_[X_train, Y_train], columns=feature_names + ['class'])
            data_frame[continuous_features] = (data_frame[continuous_features]).astype(float)
            data_frame[discrete_features] = (data_frame[discrete_features]).astype(int)

            # creating data a instance
            data = dice_ml.Data(dataframe=data_frame,
                                continuous_features=continuous_features,
                                outcome_name='class')

            # setting the pre-trained ML model for explainer
            backend = 'TF1'
            model = dice_ml.Model(model=blackbox, backend=backend)

            # creating a DiCE explainer instance
            dice_explainer = dice_ml.Dice(data, model)


            ################################### Explaining test samples #########################################
            # setting the size of the experiment
            N = 10  # number of instances to explain
            n = 5  # number of counter-factuals for every instance

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_%s_base_cfs_%s_%s.csv'%(dataset_name, N, n))
            if exists:
                os.remove(experiment_path + 'benchmark_%s_base_cfs_%s_%s.csv'%(dataset_name, N, n))
            cfs_results_csv = open(experiment_path + 'benchmark_%s_base_cfs_%s_%s.csv'%(dataset_name, N, n), 'a')

            feature_space = ['' for _ in range(X_train.shape[1]-1 + 5)]
            header = ['','MOCF']
            header += feature_space
            header += ['CFPrototype']
            header += feature_space
            header += ['DiCE']
            header = ','.join(header)
            cfs_results_csv.write('%s\n' % (header))
            cfs_results_csv.flush()

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'benchmark_%s_base_eval_%s_%s.csv'%(dataset_name, N, n))
            if exists:
                os.remove(experiment_path + 'benchmark_%s_base_eval_%s_%s.csv'%(dataset_name, N, n))
            eval_results_csv = open(experiment_path + 'benchmark_%s_base_eval_%s_%s.csv'%(dataset_name, N, n), 'a')

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('MOCF', '', '', '',
                       'CFPrototype', '', '', '',
                       'DiCE', '', '', '')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('prediction', 'distance', 'sparsity', 'diversity',
                       'prediction', 'distance', 'sparsity', 'diversity',
                       'prediction', 'distance', 'sparsity', 'diversity')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('=average(A5:A1000)', '=average(B5:B1000)',
                       '=average(C5:C1000)', '=average(D5:D1000)',
                       '=average(E5:E1000)', '=average(F5:F1000)',
                       '=average(G5:G1000)', '=average(H5:H1000)',
                       '=average(I5:I1000)', '=average(J5:J1000)',
                       '=average(K5:K1000)', '=average(L5:L1000)')
            eval_results_csv.write(header)

            header = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('=stdev(A5:A1000)', '=stdev(B5:B1000)',
                       '=stdev(C5:C1000)', '=stdev(D5:D1000)',
                       '=stdev(E5:E1000)', '=stdev(F5:F1000)',
                       '=stdev(G5:G1000)', '=stdev(H5:H1000)',
                       '=stdev(I5:I1000)', '=stdev(J5:J1000)',
                       '=stdev(K5:K1000)', '=stdev(L5:L1000)')
            eval_results_csv.write(header)
            eval_results_csv.flush()

            # selecting instances to explain from test set
            X_explain = X_test[np.random.choice(range(X_test.shape[0]), size=N, replace=False)]

            # explaining instances
            for x_ord in X_explain:

                ## MOCF explainer
                # generating counter-factuals
                mocf_explanations = mocf_explainer.explain(x_ord, cf_class= 'opposite', probability_thresh= 0.5)

                mocf_cfs_ord = mocf_explanations['cfs_ord']
                toolbox = mocf_explanations['toolbox']
                objective_names = mocf_explanations['objective_names']
                objective_weights = mocf_explanations['objective_weights']
                featureScaler = mocf_explanations['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counter-factuals
                mocf_cfs_ord,\
                mocf_cfs_eval, \
                mocf_x_cfs_ord, \
                mocf_x_cfs_eval = evaluateCounterfactuals(x_ord, mocf_cfs_ord, dataset, predict_fn,
                                                           predict_proba_fn, task, toolbox,
                                                           objective_names, objective_weights,
                                                           featureScaler, feature_names)

                # recovering counter-factuals in original format
                mocf_x_org, \
                mocf_cfs_org, \
                mocf_x_cfs_org, \
                mocf_x_cfs_highlight = recoverOriginals(x_ord, mocf_cfs_ord, dataset, feature_names)


                ## CFPrototype explainer
                # generating counter-factuals
                x_ohe = ord2ohe(x_ord, dataset)
                x_ohe = x_ohe.reshape((1,) + x_ohe.shape)
                cfprototype_explanations = cfprototype_explainer.explain(x_ohe, target_class=None)

                # extracting solutions
                cfs = []
                cfs.append(cfprototype_explanations.cf['X'].ravel())
                for iter, res in cfprototype_explanations.all.items():
                    for cf in res:
                        cfs.append(cf.ravel())
                feature_names = dataset['feature_names']
                cfs_ohe = np.asarray(cfs)
                cfprototype_cfs_ord = ohe2ord(cfs_ohe, dataset)
                cfprototype_cfs_ord = pd.DataFrame(data=cfprototype_cfs_ord, columns=feature_names)

                # evaluating counter-factuals
                cfprototype_cfs_ord, \
                cfprototype_cfs_eval, \
                cfprototype_x_cfs_ord, \
                cfprototype_x_cfs_eval = evaluateCounterfactuals(x_ord, cfprototype_cfs_ord, dataset, predict_fn,
                                                                  predict_proba_fn, task, toolbox,
                                                                  objective_names, objective_weights,
                                                                  featureScaler, feature_names)

                # recovering counter-factuals in original format
                cfprototype_x_org, \
                cfprototype_cfs_org, \
                cfprototype_x_cfs_org, \
                cfprototype_x_cfs_highlight = recoverOriginals(x_ord, cfprototype_cfs_ord, dataset, feature_names)


                ## DiCE explainer
                x_ord_dice = {}
                for key, value in zip(feature_names, list(x_ord)):
                    x_ord_dice[key] = value
                for f in discrete_features:
                    x_ord_dice[f] = str(int(x_ord_dice[f]))
                dice_explanations  = dice_explainer.generate_counterfactuals(x_ord_dice, total_CFs=n,
                                                                             desired_class="opposite",
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_algorithm="binary")

                ## extracting solutions
                dice_cfs_ord = dice_explanations.final_cfs_df.iloc[:, :-1]
                dice_cfs_ord[discrete_features] = dice_cfs_ord[discrete_features].astype(int)

                # evaluating counter-factuals
                dice_cfs_ord, \
                dice_cfs_eval, \
                dice_x_cfs_ord, \
                dice_x_cfs_eval = evaluateCounterfactuals(x_ord, dice_cfs_ord, dataset, predict_fn,
                                                         predict_proba_fn, task, toolbox,
                                                         objective_names, objective_weights,
                                                         featureScaler, feature_names)

                # recovering counter-factuals in original format
                dice_x_org, \
                dice_cfs_org, \
                dice_x_cfs_org, \
                dice_x_cfs_highlight = recoverOriginals(x_ord, dice_cfs_ord, dataset, feature_names)


                cfs_results = pd.concat([mocf_x_cfs_highlight.iloc[:n+1], mocf_x_cfs_eval.iloc[:n+1],
                                         cfprototype_x_cfs_highlight.iloc[:n+1], cfprototype_x_cfs_eval.iloc[:n+1],
                                         dice_x_cfs_highlight.iloc[:n+1], dice_x_cfs_eval.iloc[:n+1]], axis=1)
                cfs_results.to_csv(cfs_results_csv)
                cfs_results_csv.write('\n')
                cfs_results_csv.flush()

                # measuring the diversity of counter-factuals
                mocf_feature_names = []
                cfprototype_feature_names = []
                dice_feature_names = []
                for i in range(n):
                    mocf_feature_names.append([feature_names[ii] for ii in np.where(mocf_x_cfs_highlight.iloc[i+1] != '_')[0]])
                    cfprototype_feature_names.append([feature_names[ii] for ii in np.where(cfprototype_x_cfs_highlight.iloc[i + 1] != '_')[0]])
                    dice_feature_names.append([feature_names[ii] for ii in np.where(dice_x_cfs_highlight.iloc[i + 1] != '_')[0]])
                mocf_feature_names = list(filter(None, mocf_feature_names))
                cfprototype_feature_names = list(filter(None, cfprototype_feature_names))
                dice_feature_names = list(filter(None, dice_feature_names))

                mocf_jaccard = []
                cfprototype_jaccard = []
                dice_jaccard = []
                for i in range(0, n):
                    for ii in range(i, n):
                        if len(mocf_feature_names) > ii:
                            jaccard = len(set(mocf_feature_names[i]) & set(mocf_feature_names[ii])) / \
                                      len(set(mocf_feature_names[i]) | set(mocf_feature_names[ii]))
                            mocf_jaccard.append(jaccard)

                        if len(cfprototype_feature_names) > ii:
                            jaccard = len(set(cfprototype_feature_names[i]) & set(cfprototype_feature_names[ii])) / \
                                      len(set(cfprototype_feature_names[i]) | set(cfprototype_feature_names[ii]))
                            cfprototype_jaccard.append(jaccard)

                        if len(dice_feature_names) > ii:
                            jaccard = len(set(dice_feature_names[i]) & set(dice_feature_names[ii])) / \
                                      len(set(dice_feature_names[i]) | set(dice_feature_names[ii]))
                            dice_jaccard.append(jaccard)

                eval_results = np.r_[mocf_cfs_eval.iloc[:n,:-2].mean(), 1.0 - np.mean(mocf_jaccard),
                                    cfprototype_cfs_eval.iloc[:n,:-2].mean(), 1.0 - np.mean(cfprototype_jaccard),
                                    dice_cfs_eval.iloc[:n,:-2].mean(), 1.0 - np.mean(dice_jaccard)]

                eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                eval_results = ','.join(eval_results)
                eval_results_csv.write('%s\n' % (eval_results))
                eval_results_csv.flush()

            cfs_results_csv.close()
            eval_results_csv.close()

if __name__ == '__main__':
    main()
