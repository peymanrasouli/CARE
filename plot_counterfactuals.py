from utils import *
from prepare_datasets import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from create_model import CreateModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from mocf import MOCF
from evaluate_counterfactuals import evaluateCounterfactuals
from sklearn.metrics import f1_score, accuracy_score
from matplotlib.colors import ListedColormap

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'iris': (None, PrepareIris, 'classification'),
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
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            predict_fn = lambda x: blackbox.predict(x).ravel()
            predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # creating an instance of MOCF explainer for  soundCF=False and feasibleAR=False
            explainer_base = MOCF(dataset, task=task, predict_fn=predict_fn,
                                  predict_proba_fn=predict_proba_fn,
                                  soundCF=False, feasibleAR=False)
            explainer_base.fit(X_train, Y_train)

            # creating an instance of MOCF explainer for  soundCF=True and feasibleAR=False
            explainer_sound = MOCF(dataset, task=task, predict_fn=predict_fn,
                                   predict_proba_fn=predict_proba_fn,
                                   soundCF=True, feasibleAR=False)
            explainer_sound.fit(X_train, Y_train)

            pca = PCA(n_components=2)
            pca.fit(X_train)
            X_train_2d = pca.transform(X_train)
            X_test_2d = pca.transform(X_test)

            bb = GradientBoostingClassifier()
            bb.fit(X_train_2d, Y_train)
            pred_test = bb.predict(X_test_2d)
            bb_accuracy_score = accuracy_score(Y_test, pred_test)
            print('bb' , 'blackbox accuracy=', bb_accuracy_score)
            bb_f1_score = f1_score(Y_test, pred_test,average='macro')
            print('bb' , 'blackbox F1-score=', bb_f1_score)

            ################################### Explaining test samples #########################################
            # setting size of the experiment
            N = 20  # number of instances to explain

            # selecting instances to explain from test set
            np.random.seed(42)
            ind_explain = np.random.choice(range(X_test.shape[0]), size=N, replace=False)
            X_explain = X_test[ind_explain]

            # explaining instances
            for i, x_ord in enumerate(X_explain):

                explanation_base = explainer_base.explain(x_ord, cf_class='strange')
                explanation_sound = explainer_sound.explain(x_ord, cf_class='strange')

                # extracting results
                cfs_ord_base = explanation_base['cfs_ord']
                cfs_ord_sound = explanation_sound['cfs_ord']
                toolbox = explanation_sound['toolbox']
                objective_names = explanation_sound['objective_names']
                featureScaler = explanation_sound['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counter-factuals base
                cfs_ord_base, \
                cfs_eval_base, \
                x_cfs_ord_base, \
                x_cfs_eval_base = evaluateCounterfactuals(x_ord, cfs_ord_base, dataset, predict_fn, predict_proba_fn,
                                                          task, toolbox, objective_names, featureScaler, feature_names)

                # evaluating counter-factuals sound
                cfs_ord_sound, \
                cfs_eval_sound, \
                x_cfs_ord_sound, \
                x_cfs_eval_sound = evaluateCounterfactuals(x_ord, cfs_ord_sound, dataset, predict_fn, predict_proba_fn,
                                                           task, toolbox, objective_names, featureScaler, feature_names)

                cf_base_ord = explanation_base['best_cf_ord'].to_numpy()
                cf_sound_ord = explanation_sound['best_cf_ord'].to_numpy()

                x_ohe = ord2ohe(x_ord, dataset)
                cf_base_ohe = ord2ohe(cf_base_ord,dataset)
                cf_sound_ohe = ord2ohe(cf_sound_ord, dataset)

                x_class = predict_fn(x_ohe.reshape(1,-1))
                cf_base_class = predict_fn(cf_base_ohe.reshape(1,-1))
                cf_sound_class = predict_fn(cf_sound_ohe.reshape(1, -1))

                X = np.r_[pca.transform(x_ord.reshape(1,-1)),
                          pca.transform(cf_base_ord.reshape(1,-1)),
                          pca.transform(cf_sound_ord.reshape(1,-1)),
                          X_train_2d]


                y = np.r_[x_class, cf_base_class, cf_sound_class, Y_train]

                # setup marker generator and color map
                markers = ('s', 'o', 'D', '^', 'v')
                colors = ('red', 'blue', 'green', 'cyan', 'gray')

                plt.close('all')
                f = plt.figure()
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                Z = bb.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, cmap='Set2')
                plt.xlabel('D1')
                plt.ylabel('D2')


                for idx, cl in enumerate(np.unique(y)):
                    plt.scatter(X[y == cl, 0],
                                X[y == cl, 1],
                                alpha=1,
                                c=colors[idx],
                                marker=markers[idx],
                                s=30,
                                label=cl,
                                edgecolor='black')

                # highlight x
                X_x, y_x = X[0, :], y[0]
                plt.scatter(X_x[0],
                            X_x[1],
                            c='',
                            edgecolor='black',
                            alpha=1.0,
                            linewidth=2,
                            marker='D',
                            s=150,
                            label='x')

                # highlight base counter-factual
                X_cf_base, y_cf_base = X[1, :], y[1]
                plt.scatter(X_cf_base[0],
                            X_cf_base[1],
                            c='',
                            edgecolor='black',
                            alpha=1.0,
                            linewidth=2,
                            marker='o',
                            s=150,
                            label='Base CF')

                # highlight sound counter-factual
                X_cf_sound, y_cf_sound = X[2, :], y[2]
                plt.scatter(X_cf_sound[0],
                            X_cf_sound[1],
                            c='',
                            edgecolor='black',
                            alpha=1.0,
                            linewidth=2,
                            marker='s',
                            s=150,
                            label='Sound CF')
                plt.legend(loc="upper left")
                plt.title(('Base proximity= %.3f, connectedness= %.3f | Sound proximity= %.3f, connectedness= %.3f') %
                          (x_cfs_eval_base.iloc[1, 1], x_cfs_eval_base.iloc[1, 2] ,x_cfs_eval_sound.iloc[1, 1],
                           x_cfs_eval_sound.iloc[1, 2]), fontsize=9)
                plt.show()
                f.savefig(experiment_path+str(ind_explain[i])+'.pdf')
                plt.close()

                print('Done!')

if __name__ == '__main__':
    main()
