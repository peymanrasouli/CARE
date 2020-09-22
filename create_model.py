from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error

def CreateModel(X_train, X_test, Y_train, Y_test, task, model_name, constructor):

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