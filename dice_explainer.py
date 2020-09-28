import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import dice_ml
from dice_ml import model, data
from dice_ml.utils import helpers # helper functions
from mappings import BB2Theta, Theta2BB, BB2Original
from evaluate_counterfactuals import EvaluateCounterfactuals
from highlight_changes import HighlightChanges

def DiCEExplainer(x_bb, blackbox, X_train, Y_train, dataset, task, MOCF_output):

    feature_names = dataset['feature_names']
    feature_encoder = dataset['feature_encoder']
    feature_scaler = dataset['feature_scaler']
    discrete_indices = dataset['discrete_indices']
    continuous_indices = dataset['continuous_indices']
    continuous_features = [feature_names[f] for f in continuous_indices]

    feature_min = np.min(dataset['X'], axis=0)
    feature_max = np.max(dataset['X'], axis=0)
    feature_range = tuple([feature_min.reshape(1, -1), feature_max.reshape(1, -1)])


    X_train_df = pd.DataFrame(data=np.c_[X_train,Y_train], columns=dataset['df'].columns)

    d = dice_ml.Data(dataframe=X_train_df,
                     continuous_features=feature_names,
                     outcome_name=X_train_df.columns[-1])

    train, _ = d.split_data(d.normalize_data(d.one_hot_encoded_data))

    X_train = train.loc[:, train.columns != 'income']
    y_train = train.loc[:, train.columns == 'income']
    # Fitting a dense neural network model
    ann_model = keras.Sequential()
    ann_model.add(
        keras.layers.Dense(20, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l1(0.001),
                           activation=tf.nn.relu))
    ann_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    ann_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    ann_model.fit(X_train, y_train, validation_split=0.20, epochs=100, verbose=0)

    # Generate the DiCE model for explanation
    m = model.Model(model=ann_model)

    # Pre-trained ML model
    m = dice_ml.Model(model=blackbox)

    # DiCE explanation instance
    exp = dice_ml.Dice(d, m)


    # x_bb_ = x_bb.reshape(1, -1)
    # n_features = x_bb_.shape
    #
    # cat_vars = {}
    # for d in dataset['discrete_indices']:
    #     cat_vars[d] = int(feature_range[1][0,d] + 1)
    #
    # # # Prototype CF
    # prototype_cf_explainer = CounterFactualProto(predict=blackbox.predict_proba, shape=n_features,
    #                                              feature_range=feature_range, cat_vars=cat_vars, ohe=False)
    #
    # prototype_cf_explainer.fit(X_train, d_type='abdm', disc_perc=[25, 50, 75])
    #
    # explanations = prototype_cf_explainer.explain(x_bb_)
    #
    # cfs = []
    # cfs.append(explanations.cf['X'].ravel())
    # for iter, res in explanations.all.items():
    #     for cf in res:
    #         cfs.append(cf.ravel())
    #
    # cfs = np.asarray(cfs)
    # cfs = pd.DataFrame(data=cfs, columns=feature_names)
    #
    # ## Evaluating counter-factuals
    # toolbox = MOCF_output['toolbox']
    # OBJ_name = MOCF_output['OBJ_name']
    # ea_scaler = MOCF_output['ea_scaler']
    # solutions = BB2Theta(cfs, ea_scaler)
    # cfs, cfs_eval = EvaluateCounterfactuals(cfs, solutions, blackbox, toolbox, OBJ_name, task)
    #
    # ## Recovering original data
    # x_original = BB2Original(x_bb, feature_encoder, feature_scaler, discrete_indices, continuous_indices)
    # cfs_original = BB2Original(cfs, feature_encoder, feature_scaler, discrete_indices, continuous_indices)
    # cfs_original = pd.concat([x_original, cfs_original])
    # index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(cfs_original) - 1)])
    # cfs_original = cfs_original.set_index(index)
    #
    # ## Highlighting changes
    # cfs_original_highlight = HighlightChanges(cfs_original)

    # ## Returning the results
    # output = {'solutions': solutions,
    #           'cfs': cfs,
    #           'cfs_original': cfs_original,
    #           'cfs_original_highlight': cfs_original_highlight,
    #           'cfs_eval': cfs_eval,
    #           }

    return 0