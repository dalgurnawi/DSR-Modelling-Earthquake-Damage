from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
import os.path
import glob
import time


def unpickle_model(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            f.close()
        return model
    else:
        raise FileNotFoundError


def test_model(X_train, y_train, X_test, y_test, model, cv=10, ):
    unpickled_model = unpickle_model(model)
    model_prediction = unpickled_model.predict(X_test)
    macro = 'macro'
    micro = 'micro'
    print("%s CrossValidationScore: %s" % (model, cross_val_score(unpickled_model,
                                                                  X_train,
                                                                  y_train,
                                                                  cv=cv,
                                                                  verbose=0,
                                                                  n_jobs=-1)))

    print("%s with %s-averaged-F1 Score: %s " % (model, macro, f1_score(y_test,
                                                                        model_prediction,
                                                                        average=macro)))

    print("%s with %s-averaged-F1 Score: %s \n" % (model, micro, f1_score(y_test,
                                                                          model_prediction,
                                                                          average=micro)))


def evaluate_all_models(X_train, y_train, X_test, y_test, cv=10):
    total_t_start = time.time()
    model_list = glob.glob('./models/model*')
    for model in model_list:
        t_start = time.time()
        test_model(X_train, y_train, X_test, y_test, model, cv=cv)
        t_stop = time.time()
        print('Testing of a %s model took: %ss\n' % (model, t_stop - t_start))
    total_t_stop = time.time()
    print('Testing of all models took: %ss' % (total_t_stop - total_t_start))

