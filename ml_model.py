import data_preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


if __name__ == '__main__':
    print('Start data preprocessing...')
    x_train, x_val, x_test, y_train, y_val, y_test = data_preprocess.load_dataset_ml(SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)
    
    print('ml_model.py finished data preprocessing, loading, and separted them into train, val, and test sets.')
    print()
    print('================== START SVM ==================')
    print()
    range1 = [0.1 + 0.3*i for i in range(3)]
    range2 = [1 + 2*i for i in range(5)]
    #combining the ranges
    C_values = range1 + range2
    param_grid_svm = {
        'C': C_values,
        'kernel': ['rbf', 'poly'],
        'degree': [1, 2, 4]
    }
    svm_model = GridSearchCV(SVC(), param_grid=param_grid_svm, scoring='f1_macro', cv=5, verbose=2, n_jobs=1)

    svm_model.fit(x_train, y_train)

    #save the best model
    best_svm = svm_model.best_estimator_

    y_val_pred = best_svm.predict(x_val)
    y_test_pred = best_svm.predict(x_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

    print("Best Parameters SVM:", svm_model.best_params_)

    print('Validation accuracy with SVM:', val_accuracy)
    print('Validation Micro F1 Score with SVM:', val_f1_micro)
    print('Validation Macro F1 Score with SVM:', val_f1_macro)

    print('Test accuracy with SVM:', test_accuracy)
    print('Test Micro F1 Score with SVM:', test_f1_micro)
    print('Test Macro F1 Score with SVM:', test_f1_macro)
    print()
    print('================== END SVM ==================')
    print()

    # TODO: more models like random forest, regression, xgboost


    print()
    print('================== START RF ==================')
    print()
    #start random forest
    param_grid_rf = {
        'n_estimators': [10 + 2*i for i in range(5)],
        'max_depth': [5 + 2*i for i in range(8)],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, scoring='f1_macro', cv=5, verbose=2, n_jobs=1)

    rf_model.fit(x_train, y_train)

    #save the best model
    best_rf = rf_model.best_estimator_
    y_val_pred = best_rf.predict(x_val)
    y_test_pred = best_rf.predict(x_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

    print("Best Parameters RF:", rf_model.best_params_)

    print('Validation accuracy with RF:', val_accuracy)
    print('Validation Micro F1 Score with RF:', val_f1_micro)
    print('Validation Macro F1 Score with RF:', val_f1_macro)

    print('Test accuracy with RF:', test_accuracy)
    print('Test Micro F1 Score with RF:', test_f1_micro)
    print('Test Macro F1 Score with RF:', test_f1_macro)

    print()
    print('================== END RF ==================')
    print()


    print()
    print('================== START LOGR ==================')
    print()
    range1 = [0.01 + 0.03*i for i in range(3)]
    range2 = [0.1 + 0.3*i for i in range(3)]
    range3 = [1 + 5*i for i in range(5)]
    #combining the ranges
    C_values = range1 + range2 + range3
    param_grid = {
    'C': C_values,
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga', 'lbfgs']
    }
    logr_model = GridSearchCV(LogisticRegression(), param_grid, scoring='f1_macro', cv=5, verbose=2, n_jobs=1)
    logr_model.fit(x_train, y_train)

    #save the best model
    best_logr = logr_model.best_estimator_
    y_val_pred = best_logr.predict(x_val)
    y_test_pred = best_logr.predict(x_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

    print("Best Parameters LOGR:", logr_model.best_params_)

    print('Validation accuracy with LOGR:', val_accuracy)
    print('Validation Micro F1 Score with LOGR:', val_f1_micro)
    print('Validation Macro F1 Score with LOGR:', val_f1_macro)

    print('Test accuracy with LOGR:', test_accuracy)
    print('Test Micro F1 Score with LOGR:', test_f1_micro)
    print('Test Macro F1 Score with LOGR:', test_f1_macro)
    print()
    print('================== END LOGR ==================')
    print()


    print()
    print('================== START XGBOOST ==================')
    print()
    range1 = [0.01 + 0.03 * i for i in range(3)]
    range2 = [0.1 + 0.3 * i for i in range(3)]
    range3 = [1 + 5 * i for i in range(8)]

    learning_rates = range1 + range2 + range3

    param_grid_xgb = {
        'learning_rate': learning_rates,
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [50, 100, 150],
        'subsample': [0.8, 0.9, 1]
    }

    xgb_model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid_xgb, scoring='f1_macro', cv=5, verbose=2, n_jobs=1)

    xgb_model.fit(x_train, y_train)

    best_xgb = xgb_model.best_estimator_

    y_val_pred = best_xgb.predict(x_val)
    y_test_pred = best_xgb.predict(x_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

    print("Best Parameters XGB:", xgb_model.best_params_)
    print('Validation accuracy with XGB:', val_accuracy)
    print('Validation Micro F1 Score with XGB:', val_f1_micro)
    print('Validation Macro F1 Score with XGB:', val_f1_macro)
    print('Test accuracy with XGB:', test_accuracy)
    print('Test Micro F1 Score with XGB:', test_f1_micro)
    print('Test Macro F1 Score with XGB:', test_f1_macro)

    print()
    print('================== END XGBOOST ==================')

    print('WELL DONE FROM ml_model.py :)')