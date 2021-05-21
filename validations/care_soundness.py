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

            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False)
            care_config_12.fit(X_train, Y_train)

            # setting size of the experiment
            N = 20  # number of instances to explain

            # selecting instances to explain from test set
            np.random.seed(42)
            ind_explain = np.random.choice(range(X_test.shape[0]), size=N, replace=False)
            X_explain = X_test[ind_explain]

            # explaining instances
            for i, x_ord in enumerate(X_explain):

                # explaining instance x_ord using CARE with {validity} config
                explanation_config_1 = care_config_1.explain(x_ord, cf_class='strange')

                # explaining instance x_ord using CARE with {validity, soundness} config
                explanation_config_12 = care_config_12.explain(x_ord, cf_class='strange')

                # extracting results
                cfs_ord_config_1 = explanation_config_1['cfs_ord']
                cfs_ord_config_12 = explanation_config_12['cfs_ord']
                cf_config_1_ord = explanation_config_1['best_cf_ord'].to_numpy()
                cf_config_12_ord = explanation_config_12['best_cf_ord'].to_numpy()
                toolbox = explanation_config_12['toolbox']
                objective_names = explanation_config_12['objective_names']
                featureScaler = explanation_config_12['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counterfactuals of CARE with {validity} config
                cfs_ord_config_1, \
                cfs_eval_config_1, \
                x_cfs_ord_config_1, \
                x_cfs_eval_config_1 = evaluateCounterfactuals(x_ord, cfs_ord_config_1, dataset,
                                                              predict_fn, predict_proba_fn, task,
                                                              toolbox, objective_names, featureScaler,
                                                              feature_names)

                # evaluating counterfactuals of CARE with {validity, soundness} config
                cfs_ord_config_12, \
                cfs_eval_config_12, \
                x_cfs_ord_config_12, \
                x_cfs_eval_config_12 = evaluateCounterfactuals(x_ord, cfs_ord_config_12, dataset,
                                                               predict_fn, predict_proba_fn, task,
                                                               toolbox, objective_names, featureScaler,
                                                               feature_names)

                # merging counterfactuals with the training data
                x_ohe = ord2ohe(x_ord, dataset)
                cf_config_1_ohe = ord2ohe(cf_config_1_ord,dataset)
                cf_config_12_ohe = ord2ohe(cf_config_12_ord, dataset)
                X_train_ohe = ord2ohe(X_train, dataset)
                x_class = predict_fn(x_ohe.reshape(1,-1))
                cf_config_1_class = predict_fn(cf_config_1_ohe.reshape(1,-1))
                cf_config_12_class = predict_fn(cf_config_12_ohe.reshape(1, -1))
                X_train_class = predict_fn(X_train_ohe)

                X = np.r_[(x_ord.reshape(1,-1)),
                          (cf_config_1_ord.reshape(1,-1)),
                          (cf_config_12_ord.reshape(1,-1)),
                          (X_train)]
                y = np.r_[x_class, cf_config_1_class, cf_config_12_class, X_train_class]

                # setting marker generator and color map
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

                # plotting decision surface and data points
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

                # highlighting x
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

                # highlighting the counterfactual of CARE with {validity} config
                X_cf_config_1, y_cf_config_1 = X[1, :], y[1]
                plt.scatter(X_cf_config_1[0],
                            X_cf_config_1[1],
                            c='',
                            edgecolor='red',
                            alpha=1.0,
                            linewidth=2,
                            marker='o',
                            s=200,
                            label=('$\mathbf{cf_{V}}; p=%d, c=%d$') %
                             (x_cfs_eval_config_1.iloc[1, 1], x_cfs_eval_config_1.iloc[1, 2]))

                # highlighting the counterfactual of CARE with {validity, soundness} config
                X_cf_config_12, y_cf_config_12 = X[2, :], y[2]
                plt.scatter(X_cf_config_12[0],
                            X_cf_config_12[1],
                            c='',
                            edgecolor='red',
                            alpha=1.0,
                            linewidth=2,
                            marker='s',
                            s=200,
                            label=('$\mathbf{cf_{S}}; p=%d, c=%d$') %
                            (x_cfs_eval_config_12.iloc[1, 1], x_cfs_eval_config_12.iloc[1, 2]))

                plt.legend(loc=loc, handletextpad=0.1, fontsize=16)
                plt.show()
                f.savefig(experiment_path+str(ind_explain[i])+'.pdf', bbox_inches = 'tight')
                plt.close()

            print('Done!')

if __name__ == '__main__':
    main()
