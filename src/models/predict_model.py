from sklearn.metrics import f1_score
from src.data.split_data import X_test, y_test
import pickle
import os.path

def unpickle_model(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            f.close()
        return model
    else:
        raise FileNotFoundError


pipe_lr = unpickle_model('model_pipe_lr')
predictions_lr = pipe_lr.predict(X_test)
# compute logistic regression accuracy
print("Logistic Regression Accuracy with F1 Score: %s " % f1_score(y_test, predictions_lr, average='macro'))

pipe_lgbc = unpickle_model('model_pipe_lgbc')
predictions_lgbc = pipe_lgbc.predict(X_test)
# test the model
print("Gradient Boosting Accuracy with F1 Score: %s " % f1_score(y_test, predictions_lgbc, average='macro'))

pipe_cbc = unpickle_model('model_pipe_cbc')
predictions_cbc = pipe_cbc.predict(X_test)
# compute logistic regression accuracy
print("CatBoost Accuracy with F1 Score: %s " % f1_score(y_test, predictions_cbc, average='macro'))

# pipe_xgbc = unpickle_model('model_pipe_xgbc')
# predictions_xgbc = pipe_xgbc.predict(X_test)
# # test the model
# print("XGBoost Accuracy with F1 Score: %s " % f1_score(y_test, predictions_xgbc, average='macro'))

pipe_rf = unpickle_model('model_pipe_rf')
predictions_rf = pipe_rf.predict(X_test)
# test the model
print("Randomforest Accuracy with F1 Score: %s " % f1_score(y_test, predictions_rf, average='macro'))