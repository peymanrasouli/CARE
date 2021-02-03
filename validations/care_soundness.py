import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
from utils import *
from prepare_datasets import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from create_model import CreateModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from care.care import CARE
from evaluate_counterfactuals import evaluateCounterfactuals

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'moon': ('moon-sklearn', PrepareMoon, 'classification'),
        'iris': ('iris-sklearn', PrepareIris, 'classification')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'gb-c': GradientBoostingClassifier,
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path, dataset_name,usage='soundness_validation')

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            predict_fn = lambda x: blackbox.predict(x).ravel()
            predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # CARE validity
            explainer_validity = CARE(dataset, task=task, predict_fn=predict_fn,
                                  predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=False, CAUSALITY=False, ACTIONABILITY=False)
            explainer_validity.fit(X_train, Y_train)

            # CARE soundness
            explainer_soundness = CARE(dataset, task=task, predict_fn=predict_fn,
                                   predict_proba_fn=predict_proba_fn,
                                   SOUNDNESS=True, CAUSALITY=False, ACTIONABILITY=False)
            explainer_soundness.fit(X_train, Y_train)

            # setting size of the experiment
            N = 20  # number of instances to explain

            # selecting instances to explain from test set
            np.random.seed(42)
            ind_explain = np.random.choice(range(X_test.shape[0]), size=N, replace=False)
            X_explain = X_test[ind_explain]

            # explaining instances
            for i, x_ord in enumerate(X_explain):

                explanation_validity = explainer_validity.explain(x_ord, cf_class='strange')
                explanation_soundness = explainer_soundness.explain(x_ord, cf_class='strange')

                # extracting results
                cfs_ord_validity = explanation_validity['cfs_ord']
                cfs_ord_soundness = explanation_soundness['cfs_ord']
                toolbox = explanation_soundness['toolbox']
                objective_names = explanation_soundness['objective_names']
                featureScaler = explanation_soundness['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counterfactuals validity
                cfs_ord_validity, \
                cfs_eval_validity, \
                x_cfs_ord_validity, \
                x_cfs_eval_validity = evaluateCounterfactuals(x_ord, cfs_ord_validity, dataset, predict_fn, predict_proba_fn,
                                                          task, toolbox, objective_names, featureScaler, feature_names)

                # evaluating counterfactuals soundness
                cfs_ord_soundness, \
                cfs_eval_soundness, \
                x_cfs_ord_soundness, \
                x_cfs_eval_soundness = evaluateCounterfactuals(x_ord, cfs_ord_soundness, dataset, predict_fn, predict_proba_fn,
                                                           task, toolbox, objective_names, featureScaler, feature_names)

                cf_validity_ord = explanation_validity['best_cf_ord'].to_numpy()
                cf_soundness_ord = explanation_soundness['best_cf_ord'].to_numpy()

                x_ohe = ord2ohe(x_ord, dataset)
                cf_validity_ohe = ord2ohe(cf_validity_ord,dataset)
                cf_soundness_ohe = ord2ohe(cf_soundness_ord, dataset)
                X_train_ohe = ord2ohe(X_train, dataset)
                x_class = predict_fn(x_ohe.reshape(1,-1))
                cf_validity_class = predict_fn(cf_validity_ohe.reshape(1,-1))
                cf_soundness_class = predict_fn(cf_soundness_ohe.reshape(1, -1))
                X_train_class = predict_fn(X_train_ohe)

                # merging counterfactuals with the training data
                X = np.r_[(x_ord.reshape(1,-1)),
                          (cf_validity_ord.reshape(1,-1)),
                          (cf_soundness_ord.reshape(1,-1)),
                          (X_train)]
                y = np.r_[x_class, cf_validity_class, cf_soundness_class, X_train_class]

                # setup marker generator and color map
                if dataset_kw == 'iris':
                    markers = ('s', 'o', 'D')
                    colors = ('#c060a1', '#6a097d', '#f1d4d4')
                    cmap = ListedColormap(['#f1d4d4', '#c060a1', '#6a097d'])
                    loc = "lower right"
                else:
                    markers = ('s', 'D')
                    colors = ('#c060a1', '#f1d4d4')
                    cmap = ListedColormap(['#f1d4d4', '#6a097d'])
                    loc = "lower left"

                x_min, x_max = X[:, 0].min() - 1 , X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1 , X[:, 1].max() + 1
                h = 0.02
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                X_surface_ohe = ord2ohe(np.c_[xx.ravel(), yy.ravel()], dataset)
                Z = predict_fn(X_surface_ohe)
                Z = Z.reshape(xx.shape)

                # plot decision surface and data points
                plt.close('all')
                plt.rcParams['font.size'] = '16'
                f = plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                plt.contourf(xx, yy, Z, cmap=cmap)
                plt.xlabel(dataset['feature_names'][0])
                plt.ylabel(dataset['feature_names'][1])

                for idx, cl in enumerate(np.unique(y)):
                    plt.scatter(X[y == cl, 0],
                                X[y == cl, 1],
                                alpha=1,
                                c=colors[idx],
                                marker=markers[idx],
                                s=50,
                                # label=cl,
                                edgecolor='black')

                # highlight x
                X_x, y_x = X[0, :], y[0]
                plt.scatter(X_x[0],
                            X_x[1],
                            c='',
                            edgecolor='red',
                            alpha=1.0,
                            linewidth=2,
                            marker='D',
                            s=200,
                            label='$\mathbf{x}$')

                # highlight validity's counterfactual
                X_cf_validity, y_cf_validity = X[1, :], y[1]
                plt.scatter(X_cf_validity[0],
                            X_cf_validity[1],
                            c='',
                            edgecolor='red',
                            alpha=1.0,
                            linewidth=2,
                            marker='o',
                            s=200,
                            label=('$\mathbf{cf_{V}}; p=%d, c=%d$') %
                             (x_cfs_eval_validity.iloc[1, 1], x_cfs_eval_validity.iloc[1, 2]))

                # highlight soundness's counterfactual
                X_cf_soundness, y_cf_soundness = X[2, :], y[2]
                plt.scatter(X_cf_soundness[0],
                            X_cf_soundness[1],
                            c='',
                            edgecolor='red',
                            alpha=1.0,
                            linewidth=2,
                            marker='s',
                            s=200,
                            label=('$\mathbf{cf_{S}}; p=%d, c=%d$') %
                            (x_cfs_eval_soundness.iloc[1, 1], x_cfs_eval_soundness.iloc[1, 2]))

                plt.legend(loc=loc, handletextpad=0.1, fontsize=16)
                plt.show()
                f.savefig(experiment_path+str(ind_explain[i])+'.pdf', bbox_inches = 'tight')
                plt.close()

            print('Done!')

if __name__ == '__main__':
    main()
