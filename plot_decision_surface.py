import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

class PlotDecisionSurface():
    def __init__(self,
                 X_train,
                 Y_train,
                 h = 0.02):
        self.X_train = X_train
        self.Y_train = Y_train
        self.h = h

        self.pca = PCA(n_components=2)
        self.pca.fit(X_train)

        self.X_train_2d = self.pca.transform(X_train)
        self.model = GradientBoostingClassifier()
        self.model.fit(self.X_train_2d, Y_train)

    def makeMeshgrid(self, x, y):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h), np.arange(y_min, y_max, self.h))
        return xx, yy

    def plotContours(self, ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot(self, x, method):

        x_2d = self.pca.transform(x)
        X_new = np.r_[x_2d, self.X_train_2d]
        Y_new = np.r_[-1, self.Y_train]

        fig, ax = plt.subplots()
        X0, X1 = X_new[:, 0], X_new[:, 1]
        xx, yy = self.makeMeshgrid(X0, X1)

        self.plotContours(ax, self.model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=Y_new, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel('D1')
        ax.set_xlabel('D2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Decision surface for ' + method)
        ax.legend()
        plt.show()


