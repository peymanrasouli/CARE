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
from user_preferences import userPreferences
from care.care import CARE
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
import dice_ml
from certifai.certifai import CERTIFAI
from care_explainer import CAREExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer
from certifai_explainer import CERTIFAIExplainer
from generate_text_explanations import GenerateTextExplanations

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'


    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdultReduced, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
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

            # creating/opening a csv file for storing results
            exists = os.path.isfile(experiment_path + 'user_study_assessment.csv')
            if exists:
                os.remove(experiment_path + 'user_study_assessment.csv')
            user_study_csv = open(experiment_path + 'user_study_assessment.csv', 'a')


            # explaining instances from test set
            N = 50
            explained = 0
            for x_ord in X_test:

                try:

                    # setting user preferences
                    print('\n')
                    print('----- user-specified constraints -----')
                    constraints = {'age': ('ge', 1),
                                   'sex': ('fix', 1),
                                   'native-country': ('fix', 1)}

                    constraint = [None] * len(x_ord)
                    importance = [None] * len(x_ord)
                    for p in constraints:
                        index = dataset['feature_names'].index(p)
                        constraint[index] = constraints[p][0]
                        importance[index] = constraints[p][1]
                        print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')

                    user_preferences = {'constraint': constraint,
                                        'importance': importance}

                    # explain instance x_ord using CARE
                    CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                                explainer=None, SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True,
                                                user_preferences=user_preferences, cf_class='opposite',
                                                probability_thresh=0.5, n_cf=1)
                    care_x_cfs_highlight = CARE_output['x_cfs_highlight']
                    _, care_text_explanation = GenerateTextExplanations(CARE_output, dataset)
                    if int(CARE_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CARE_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using CFPrototype
                    CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset,
                                                              task, CARE_output, explainer=None,
                                                              target_class=None, n_cf=1)
                    cfprototype_x_cfs_highlight = CFPrototype_output['x_cfs_highlight']
                    _, cfprototype_text_explanation = GenerateTextExplanations(CFPrototype_output, dataset)
                    if int(CFPrototype_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CFPrototype_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using DiCE
                    DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train,
                                                dataset, task, CARE_output, explainer=None, ACTIONABILITY=True,
                                                desired_class="opposite", n_cf=1, probability_thresh=0.5,
                                                user_preferences=user_preferences,
                                                proximity_weight=1.0, diversity_weight=1.0)
                    dice_x_cfs_highlight = DiCE_output['x_cfs_highlight']
                    _, dice_text_explanation = GenerateTextExplanations(DiCE_output, dataset)
                    if int(DiCE_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(DiCE_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # explain instance x_ord using CERTIFAI
                    CERTIFAI_output = CERTIFAIExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn,
                                                        predict_proba_fn, CARE_output, explainer=None,
                                                        ACTIONABILITY=True, user_preferences=user_preferences,
                                                        cf_class='opposite', n_cf=1)
                    certifai_x_cfs_highlight = CERTIFAI_output['x_cfs_highlight']
                    _, certifai_text_explanation = GenerateTextExplanations(CERTIFAI_output, dataset)
                    if int(CERTIFAI_output['x_cfs_eval']['Class'].loc['x']) == \
                            int(CERTIFAI_output['x_cfs_eval']['Class'].loc['cf_0']):
                        raise Exception

                    # storing the best counterfactual found by methods
                    x_class = dataset['labels'][int(CARE_output['x_cfs_eval']['Class'].loc['x'])]
                    cf_class = dataset['labels'][int(CARE_output['x_cfs_eval']['Class'].loc['cf_0'])]
                    cfs_results = np.c_[np.r_[np.asarray(care_x_cfs_highlight.iloc[0, :]), np.asarray(x_class)],
                                        np.r_[np.asarray(care_x_cfs_highlight.iloc[1, :]), np.asarray(cf_class)],
                                        np.r_[np.asarray(cfprototype_x_cfs_highlight.iloc[1, :]), np.asarray(cf_class)],
                                        np.r_[np.asarray(dice_x_cfs_highlight.iloc[1, :]), np.asarray(cf_class)],
                                        np.r_[np.asarray(certifai_x_cfs_highlight.iloc[1,: ]), np.asarray(cf_class)]]

                    user_study_csv.write('Individual #%d\n' % (explained))
                    user_study_csv.flush()

                    cfs_results_df = pd.DataFrame(columns=['Original', 'CE 1', 'CE 2', 'CE 3', 'CE 4'],
                                                  data= cfs_results,
                                                  index= dataset['feature_names']+['Class']).transpose()
                    cfs_results_df.to_csv(user_study_csv, sep='\t')
                    user_study_csv.write('\n')
                    user_study_csv.flush()

                    actionability = ['Actionable', '', '', '', '']
                    consistency = ['Consistent', '', '', '', '']
                    understandability = ['Understandable', '', '', '', '']
                    metrics = np.c_[np.asarray(actionability),np.asarray(consistency), np.asarray(understandability)].T

                    user_study_csv.write('\nMetrics\n')
                    user_study_csv.flush()
                    metrics_df = pd.DataFrame(columns=['', 'Least', '', '', 'Most'], data=metrics)
                    metrics_df.to_csv(user_study_csv, sep='\t', index=False)
                    user_study_csv.write('\n\n\n')
                    user_study_csv.flush()

                    explained += 1

                    print('-----------------------------------------------------------------------')
                    print("%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                    print('-----------------------------------------------------------------------')

                except Exception:
                    pass

                if explained == N:
                    break

            user_study_csv.close()

if __name__ == '__main__':
    main()
