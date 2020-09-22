import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel(40)
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error

def CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, model_name, constructor=None):

    if constructor is None:

        # discrete_vars = dataset['discrete_indices']
        # continuous_vars = dataset['continuous_indices']
        # n_features = X_train.shape[1]
        #
        # x_in = Input(shape=(n_features,))
        # models = []
        # for f in range(n_features):
        #     if f in discrete_vars:
        #         emb_in = Lambda(lambda x: x[:, f:f + 1])(x_in)
        #         no_of_unique_cat = len(np.unique(X_train[:, f]))
        #         embedding_size = min(np.ceil((no_of_unique_cat) / 2), 50)
        #         embedding_size = int(embedding_size)
        #         vocab = no_of_unique_cat + 1
        #         model = Embedding(vocab, embedding_size, input_length=1)(emb_in)
        #         model = Reshape(target_shape=(embedding_size,))(model)
        #         models.append(model)
        #     if f in continuous_vars:
        #         num_in = Lambda(lambda x: x[:, f:f + 1])(x_in)
        #         model = Dense(1, input_dim=1)(num_in)
        #         models.append(model)
        #
        # x = Concatenate()(models)
        # x = Dense(60, activation='relu')(x)
        # x = Dropout(.2)(x)
        # x = Dense(60, activation='relu')(x)
        # x = Dropout(.2)(x)
        # x = Dense(60, activation='relu')(x)
        # x = Dropout(.2)(x)
        # x_out = Dense(2, activation='softmax')(x)
        #
        # blackbox = Model(inputs=x_in, outputs=x_out)
        # blackbox.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # blackbox.summary()
        # blackbox.fit(X_train, to_categorical(Y_train), epochs=30)
        # pred_test = np.argmax(blackbox.predict(X_test), axis=1)
        # bb_accuracy_score = accuracy_score(Y_test, pred_test)
        # print('blackbox accuracy=', bb_accuracy_score)
        # bb_f1_score = f1_score(Y_test, pred_test,average='macro')
        # print('blackbox F1-score=', bb_f1_score)
        pass
    else:
        if task is 'classification':
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

# ## Creating and training black-box
#
#
# feature_min = np.min(dataset['X'], axis=0)
# feature_max = np.max(dataset['X'], axis=0)
# feature_range = [feature_min, feature_max]
# categorical_vars = {}
# for d in dataset['discrete_indices']:
#     categorical_vars[d] = int(feature_range[1][d] + 1)
# shape = X_train.shape[1]
# blackbox = nn_ord(categorical_vars, shape)
# blackbox.summary()
# blackbox.fit(X_train, to_categorical(Y_train), batch_size=64, epochs=30, verbose=0)
# # pred_test = blackbox.predict(X_test)
# # bb_accuracy_score = accuracy_score(Y_test, pred_test)
# # print('blackbox accuracy=', bb_accuracy_score)
# # bb_f1_score = f1_score(Y_test, pred_test,average='macro')
# # print('blackbox F1-score=', bb_f1_score)
# # print('\n')
#
#
# ## Creating and training black-box
#
# from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Reshape, Dropout, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
#
# def nn_ord(categorical_vars, shape):
#
#     x_in = Input(shape=(shape,))
#     layers_in = []
#
#     # embedding layers
#     for i, (_, v) in enumerate(categorical_vars.items()):
#         emb_in = Lambda(lambda x: x[:, i:i+1])(x_in)
#         emb_dim = int(max(min(np.ceil(.5 * v), 50), 2))
#         emb_layer = Embedding(input_dim=v+1, output_dim=emb_dim, input_length=1)(emb_in)
#         emb_layer = Reshape(target_shape=(emb_dim,))(emb_layer)
#         layers_in.append(emb_layer)
#
#     # numerical layers
#     num_in = Lambda(lambda x: x[:, -4:])(x_in)
#     num_layer = Dense(16)(num_in)
#     layers_in.append(num_layer)
#
#     # combine
#     x = Concatenate()(layers_in)
#     x = Dense(60, activation='relu')(x)
#     x = Dropout(.2)(x)
#     x = Dense(60, activation='relu')(x)
#     x = Dropout(.2)(x)
#     x = Dense(60, activation='relu')(x)
#     x = Dropout(.2)(x)
#     x_out = Dense(2, activation='softmax')(x)
#
#     nn = Model(inputs=x_in, outputs=x_out)
#     nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     return nn