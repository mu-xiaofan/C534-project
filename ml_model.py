import data_preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test = data_preprocess.load_dataset_ml(SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)

    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)

    y_val_pred = svm_model.predict(x_val)
    y_test_pred = svm_model.predict(x_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

    print('Validation accuracy with SVM:', val_accuracy)
    print('Validation Micro F1 Score with SVM:', val_f1_micro)
    print('Validation Macro F1 Score with SVM:', val_f1_macro)

    print('Test accuracy with SVM:', test_accuracy)
    print('Test Micro F1 Score with SVM:', test_f1_micro)
    print('Test Macro F1 Score with SVM:', test_f1_macro)
    # TODO: more models like random forest, regression, xgboost