import sys
sys.path.insert(0, "alibi")
sys.path.insert(0, "DiCE")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from user_preferences import userPreferences
from care_explainer import CAREExplainer
from cfprototype_explainer import CFPrototypeExplainer
from dice_explainer import DiCEExplainer
from certifai_explainer import CERTIFAIExplainer
from generate_text_explanations import GenerateTextExplanations

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPAS, 'classification'),
        # 'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'heloc': ('heloc_dataset_v1.csv', PrepareHELOC, 'classification'),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
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

            # instance to explain
            ind = 0
            x_ord = X_test[ind]
            n_cf = 10

            # set user preferences || they are taken into account when ACTIONABILITY=True!
            user_preferences = userPreferences(dataset, x_ord)

            # explain instance x_ord using CARE
            CARE_output = CAREExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                        SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False,
                                        user_preferences=user_preferences, cf_class='opposite',
                                        probability_thresh=0.5, n_cf=n_cf)

            # explain instance x_ord using CFPrototype
            CFPrototype_output = CFPrototypeExplainer(x_ord, predict_fn, predict_proba_fn, X_train, dataset, task,
                                                      CARE_output, target_class=None, n_cf=n_cf)

            # explain instance x_ord using DiCE
            DiCE_output = DiCEExplainer(x_ord, blackbox, predict_fn, predict_proba_fn, X_train, Y_train, dataset,
                                        task, CARE_output, ACTIONABILITY=False, user_preferences=user_preferences,
                                        n_cf=n_cf, desired_class="opposite", probability_thresh=0.5,
                                        proximity_weight=1.0, diversity_weight=1.0)

            # explain instance x_ord using CERTIFAI
            CERTIFAI_output = CERTIFAIExplainer(x_ord, X_train, Y_train, dataset, task, predict_fn, predict_proba_fn,
                                                CARE_output, ACTIONABILITY=False, user_preferences=user_preferences,
                                                cf_class='opposite', n_cf=n_cf)

            # print counterfactuals and their corresponding objective values
            print('\n')
            print('CARE counterfactuals')
            print(CARE_output['x_cfs_highlight'])
            print(CARE_output['x_cfs_eval'])

            print('\n')
            print('CFPrototype counterfactuals')
            print(CFPrototype_output['x_cfs_highlight'])
            print(CFPrototype_output['x_cfs_eval'])

            print('\n')
            print('DiCE counterfactuals')
            print(DiCE_output['x_cfs_highlight'])
            print(DiCE_output['x_cfs_eval'])

            print('\n')
            print('CERTIFAI counterfactuals')
            print(CERTIFAI_output['x_cfs_highlight'])
            print(CERTIFAI_output['x_cfs_eval'])

            # generate text explanations
            print('\n')
            print('CARE text explanation')
            input, text_explanation = GenerateTextExplanations(CARE_output, dataset)
            print(input, '\n \n', text_explanation)

            print('\n')
            print('CFPrototype text explanation')
            input, text_explanation = GenerateTextExplanations(CFPrototype_output, dataset)
            print(input, '\n \n', text_explanation)

            print('\n')
            print('DiCE text explanation')
            input, text_explanation = GenerateTextExplanations(DiCE_output, dataset)
            print(input, '\n \n', text_explanation)

            print('\n')
            print('CERTIFAI text explanation')
            input, text_explanation = GenerateTextExplanations(CERTIFAI_output, dataset)
            print(input, '\n \n', text_explanation)

            print('\n')
            print('Done!')

if __name__ == '__main__':
    main()
