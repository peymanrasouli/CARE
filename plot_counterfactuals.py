from utils import *
from prepare_datasets import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from create_model import CreateModel, KerasNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from mocf import MOCF

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        # 'nn-c': KerasNeuralNetwork,
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
            if blackbox_name == 'nn-c':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1 - blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # setting size of the experiment
            N = 10  # number of instances to explain

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
            X_2d = pca.transform(X_train)


            ################################### Explaining test samples #########################################

            # selecting instances to explain from test set
            np.random.seed(42)
            ind_explain = np.random.choice(range(X_test.shape[0]), size=N, replace=False)
            X_explain = X_test[ind_explain]

            # explaining instances
            for i, x_ord in enumerate(X_explain):

                explanation_base = explainer_base.explain(x_ord)
                explanation_sound = explainer_sound.explain(x_ord)

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
                          X_2d]

                y = np.r_[x_class, cf_base_class, cf_sound_class, Y_train]

                # setup marker generator and color map
                markers = ('s', 'o', 'D', '^', 'v')
                colors = ('red', 'blue', 'green', 'cyan', 'gray')

                f = plt.figure()
                for idx, cl in enumerate(np.unique(y)):
                    plt.scatter(x=X[y == cl, 0],
                                y=X[y == cl, 1],
                                alpha=1,
                                c=colors[idx],
                                marker=markers[idx],
                                s=20,
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
                            s=100,
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
                            s=100,
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
                            s=100,
                            label='Sound CF')
                plt.legend(loc="upper left")
                f.savefig(experiment_path+str(ind_explain[i])+'.pdf')
                plt.close()

                print('Done!')

if __name__ == '__main__':
    main()
