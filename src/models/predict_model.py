from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from src.data.split_data import X_test, y_test
import pickle
import os.path
import glob


def unpickle_model(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            f.close()
        return model
    else:
        raise FileNotFoundError


def test_model(model):
    unpickled_model = unpickle_model(model)
    model_prediction = unpickled_model.predict(X_test)
    macro = 'macro'
    micro = 'micro'
    # print("Classification report: \n", classification_report(model_prediction, y_test))
    # print("Confusion matrix: \n", confusion_matrix(model_prediction, y_test))
    # print("Accuracy score: %s" % accuracy_score(model_prediction, y_test))
    # print("Model score: %s" % unpickled_model.score(X_test, y_test))
    print("%s with %s-averaged-F1 Score: %s " % (model, macro, f1_score(y_test, model_prediction, average=macro)))
    print("%s with %s-averaged-F1 Score: %s \n" % (model, micro, f1_score(y_test, model_prediction, average=micro)))


def test_all_models():
    model_list = glob.glob('model*')
    for model in model_list:
        test_model(model)


# X_test = X_test.head(1000)
# y_test = y_test.head(1000)
test_all_models()

# pipe_lr = unpickle_model('model_pipe_lr')
# predictions_lr = pipe_lr.predict(X_test)
# # compute logistic regression accuracy
# print("Logistic Regression Accuracy with F1 Score: %s " % f1_score(y_test, predictions_lr, average='macro'))
#
# pipe_lgbc = unpickle_model('model_pipe_lgbc')
# predictions_lgbc = pipe_lgbc.predict(X_test)
# # test the model
# print("Gradient Boosting Accuracy with F1 Score: %s " % f1_score(y_test, predictions_lgbc, average='macro'))
#
# pipe_cbc = unpickle_model('model_pipe_cbc')
# predictions_cbc = pipe_cbc.predict(X_test)
# # compute logistic regression accuracy
# print("CatBoost Accuracy with F1 Score: %s " % f1_score(y_test, predictions_cbc, average='macro'))
#
# # pipe_xgbc = unpickle_model('model_pipe_xgbc')
# # predictions_xgbc = pipe_xgbc.predict(X_test)
# # # test the model
# # print("XGBoost Accuracy with F1 Score: %s " % f1_score(y_test, predictions_xgbc, average='macro'))
#
# pipe_rf = unpickle_model('model_pipe_rf')
# predictions_rf = pipe_rf.predict(X_test)
# # test the model
# print("Randomforest Accuracy with F1 Score: %s " % f1_score(y_test, predictions_rf, average='macro'))
