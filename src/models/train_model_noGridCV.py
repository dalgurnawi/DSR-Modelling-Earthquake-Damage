from src.data.split_data import X_train, y_train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import pickle
import time

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# Accuracy score is the simplest way to evaluate
# print(accuracy_score(SVC_prediction, y_test))
# print(accuracy_score(KNN_prediction, y_test))
# # But Confusion Matrix and Classification Report give more details about performance
# print(confusion_matrix(SVC_prediction, y_test))
# print(classification_report(KNN_prediction, y_test)

# >>> from sklearn.model_selection import cross_val_score
# >>> from sklearn.ensemble import ExtraTreesClassifier
# >>> from sklearn.ensemble import GradientBoostingClassifier
# https://scikit-learn.org/stable/modules/ensemble.html

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ???from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# >>> from sklearn.ensemble import BaggingClassifier

lr_param = {'class_weight': 'balanced', 'max_iter': 100, 'n_jobs': -1, 'penalty': 'l1', 'solver': 'liblinear'}
lgbm_parameters = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 50}
rf_parameters = {'criterion': 'gini', 'max_depth': 15, 'n_estimators': 1500, 'n_jobs': -1}
noparam = {'n_jobs': -1}

all_models = [
    ("AdaBoost", AdaBoostClassifier()),
    ("CatBoost", cb.CatBoostClassifier()),
    ("Decision_Tree", DecisionTreeClassifier(max_depth=5)),
    ("Gaussian_Process", GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1)),
    ("LGBCM(with_param)",lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)),
    ("LGBMC(no_param)", lgb.LGBMClassifier(n_jobs=-1)),
    ("Linear_SVM", SVC(kernel="linear", C=0.025)),
    ("Logistic_Regression_Vanilla", LogisticRegression(n_jobs=-1)),
    ("Logistic_Regression_Parametrized", LogisticRegression(n_jobs=-1)),
    ("Naive_Bayes", GaussianNB()),
    ("Nearest_Neighbors", KNeighborsClassifier(3, n_jobs=-1)),
    ("Neural_Net", MLPClassifier(alpha=1, max_iter=1000)),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Random_Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1)),
    ("RBF_SVM", SVC(gamma=2, C=1)),
    ("XGBoost", xgb.XGBClassifier())
]

baseline_models_list = [
    ("Logistic_Regression_Vanilla", LogisticRegression, noparam),
    ("Logistic_Regression(with_param)", LogisticRegression, lr_param),
    ("Random_Forest(with_param)", RandomForestClassifier, rf_parameters),
    ("Random_Forest(no_param)", RandomForestClassifier, noparam),
    ("LGBCM(with_param)", lgb.LGBMClassifier, lgbm_parameters),
    ("LGBMC(no_param)", lgb.LGBMClassifier, noparam)
]


def pickle_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        f.close()


def train_and_pickle_model(clf, name, parameters):
    init_model = clf(**parameters)
    model = init_model.fit(X_train, y_train)
    pickle_model(model, 'model_%s' % name)


def train_all_models(models_list):
    total_t_start = time.time()
    for model_set in models_list:
        t_start = time.time()
        train_and_pickle_model(model_set[1], model_set[0], model_set[2])
        t_stop = time.time()
        print('Training of a %s model took: %ss' % (model_set[0], t_stop - t_start))
    total_t_stop = time.time()
    print('Training of all model took: %ss' % (total_t_stop - total_t_start))


train_all_models(baseline_models_list)