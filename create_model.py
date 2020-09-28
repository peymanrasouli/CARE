import numpy as np
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow import keras
# import tensorflow as tf

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error

def CreateModel(X_train, X_test, Y_train, Y_test, task, model_name, constructor):

    if task is 'classification':
        if model_name is 'dnn':
            # class Blackbox:
            #     def __init__(self, X_train, Y_train):
            #         n_features = X_train.shape[1]
            #         model = Sequential()
            #         model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
            #         # blackbox.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
            #         model.add(Dense(1, activation='sigmoid'))
            #         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            #         model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)
            #         self.model = model
            #     def predict(self, X):
            #         classes = self.model.predict_classes(X).ravel()
            #         return classes
            #     def predict_proba(self, X):
            #         prob_class_1 = self.model.predict(X)
            #         prob_class_0 = 1 - prob_class_1
            #         probabilities = np.c_[prob_class_0,prob_class_1]
            #         return probabilities
            #
            # blackbox = Blackbox(X_train,Y_train)
            # pred_test = blackbox.predict(X_test)
            # bb_accuracy_score = accuracy_score(Y_test, pred_test)
            # print(model_name , 'blackbox accuracy=', bb_accuracy_score)
            # bb_f1_score = f1_score(Y_test, pred_test,average='macro')
            # print(model_name , 'blackbox F1-score=', bb_f1_score)

            # ann_model = keras.Sequential()
            # ann_model.add(keras.layers.Dense(20, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l1(0.001), activation=tf.nn.relu))
            # ann_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
            # ann_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
            # ann_model.fit(X_train, Y_train, validation_split=0.20, epochs=20, verbose=1, class_weight={0: 1, 1: 2})

            # return ann_model

            # n_features = X_train.shape[1]
            # model = Sequential()
            # model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
            # # blackbox.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
            # model.add(Dense(1, activation='sigmoid'))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)
            #
            # return model

            return 0


        else:
            blackbox = constructor(random_state=42)
            blackbox.fit(X_train, Y_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy_score = accuracy_score(Y_test, pred_test)
            print(model_name , 'blackbox accuracy=', bb_accuracy_score)
            bb_f1_score = f1_score(Y_test, pred_test,average='macro')
            print(model_name , 'blackbox F1-score=', bb_f1_score)
            return blackbox

    elif task is 'regression':
        blackbox = constructor(random_state=42)
        blackbox.fit(X_train, Y_train)
        pred_test = blackbox.predict(X_test)
        bb_mae_error = mean_absolute_error(Y_test, pred_test)
        print(model_name , 'blackbox MAE=', bb_mae_error)
        bb_mse_error = mean_squared_error(Y_test, pred_test)
        print(model_name , 'blackbox MSE=', bb_mse_error)
        return blackbox