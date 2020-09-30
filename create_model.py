from mappings import ord2ohe
import tensorflow as tf
tf.InteractiveSession()
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error

def CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, model_name, constructor):
    X_train_ohe = ord2ohe(X_train, dataset)
    X_test_ohe = ord2ohe(X_test, dataset)

    if task is 'classification':
        if model_name is 'dnn':
            # Initialize variables in the graph/model
            blackbox = keras.Sequential()
            blackbox.add(keras.layers.Dense(10, input_shape=(X_train_ohe.shape[1],),
                                            kernel_regularizer=keras.regularizers.l1(0.001), activation=tf.nn.relu))
            blackbox.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
            blackbox.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
            blackbox.fit(X_train_ohe, Y_train, validation_split=0.20, epochs=20, verbose=1)
            pred_test = (blackbox.predict(X_test_ohe) > 0.5).astype("int32")
            bb_accuracy_score = accuracy_score(Y_test, pred_test)
            print(model_name , 'blackbox accuracy=', bb_accuracy_score)
            bb_f1_score = f1_score(Y_test, pred_test,average='macro')
            print(model_name , 'blackbox F1-score=', bb_f1_score)
            return blackbox
        else:
            blackbox = constructor(random_state=42)
            blackbox.fit(X_train_ohe, Y_train)
            pred_test = blackbox.predict(X_test_ohe)
            bb_accuracy_score = accuracy_score(Y_test, pred_test)
            print(model_name , 'blackbox accuracy=', bb_accuracy_score)
            bb_f1_score = f1_score(Y_test, pred_test,average='macro')
            print(model_name , 'blackbox F1-score=', bb_f1_score)
            return blackbox

    elif task is 'regression':
        blackbox = constructor(random_state=42)
        blackbox.fit(X_train_ohe, Y_train)
        pred_test = blackbox.predict(X_test_ohe)
        bb_mae_error = mean_absolute_error(Y_test, pred_test)
        print(model_name , 'blackbox MAE=', bb_mae_error)
        bb_mse_error = mean_squared_error(Y_test, pred_test)
        print(model_name , 'blackbox MSE=', bb_mse_error)
        return blackbox