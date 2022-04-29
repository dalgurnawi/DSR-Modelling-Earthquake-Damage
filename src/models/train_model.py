from src.data.split_data import X_train, X_test, y_train, y_test
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
# >>> from sklearn.ensemble import RandomForestClassifier
# >>> from sklearn.ensemble import ExtraTreesClassifier
# >>> from sklearn.tree import DecisionTreeClassifier

# SGDClassifier which with loss=log runs Logistic Regression with Stochastic Gradient Descent optimization, and it has partial_fit method.

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

names = [
    "AdaBoost",
    "Decision_Tree",
    "CatBoost",
    "Gaussian_Process",
    "LGBCM(with_param)",
    "LGBMC(no_param)",
    "Linear_SVM",
    "Naive_Bayes",
    "Nearest_Neighbors",
    "Neural_Net",
    "QDA",
    "Random_Forest",
    "RBF_SVM",
    "XGBoost"
]

classifiers = [
    AdaBoostClassifier(),
    cb.CatBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
    lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1),
    lgb.LGBMClassifier(n_jobs=-1),
    SVC(kernel="linear", C=0.025),
    GaussianNB(),
    KNeighborsClassifier(3, n_jobs=-1),
    MLPClassifier(alpha=1, max_iter=1000),
    QuadraticDiscriminantAnalysis(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
    SVC(gamma=2, C=1),
    xgb.XGBClassifier()
]


def pickle_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        f.close()


def train_and_pickle_model(clf, name):
    model = clf.fit(X_train, y_train)
    pickle_model(model, 'model_%s' % name)


def train_all_models(model_names, model_classifiers):
    for name, clf in zip(model_names, model_classifiers):
        t_start = time.time()
        train_and_pickle_model(clf, name)
        t_stop = time.time()
        print('Training of a %s model took: %ss' % (name, t_stop - t_start))
        # score = clf.score(X_test, y_test)


train_all_models(names, classifiers)

# print("logisticregression start")
# # Initiate and train baseline LogisticRegression
# lr = LogisticRegression(n_jobs=-1)
# # train the initiated model
# pipe_lr = Pipeline([('logistic_regression', lr)])
# pipe_lr.fit(X_train, y_train)
# pickle_model(pipe_lr, 'model_pipe_lr')
# print("logisticregression stop")
#
# print("lgbc start")
# # Initiate and train LGBMClassifier
# # lgbc_parameters = {'learning_rate' : [0.1, 0.05], 'n_estimators' : [1000, 1100], 'max_depth' : [5, 6], 'num_leaves' : [50, 60]}
# # list of parameters in range for gridsearch [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# grid_lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# # lgbc = lgb.LGBMClassifier(n_jobs=-1)
# # grid_lgbc = GridSearchCV(lgbc, lgbc_parameters, refit=True, verbose=3, n_jobs=-1)
# pipe_gb = Pipeline([('gradient_boosting', grid_lgbc)])
# # fit the model
# pipe_gb.fit(X_train, y_train)
# pickle_model(pipe_gb, 'model_pipe_lgbc')
# print("lgbc stop")
#
# print("catboost start")
# # Initiate and train CatBoostClassifier
# # cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01],
#                     # 'n_estimators' : [1000, 1500, 2000],
#                     # 'max_depth' : [5, 6, 7, 8, 9 ,10],
#                     # 'num_leaves' : [50, 60, 70, 80, 90, 100]
# # }
# # lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# cbc = cb.CatBoostClassifier()
# # grid_cb = GridSearchCV(cbc, cb_parameters)
# pipe_cbc = Pipeline([('cat_boosting', cbc)])
# # fit the model
# pipe_cbc.fit(X_train, y_train)
# pickle_model(pipe_cbc, 'model_pipe_cbc')
# print("catboost stop")
#
# print("randomforest start")
# # Initiate and train CatBoostClassifier
# # cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01],
#                     # 'n_estimators' : [1000, 1500, 2000],
#                     # 'max_depth' : [5, 6, 7, 8, 9 ,10],
#                     # 'num_leaves' : [50, 60, 70, 80, 90, 100]
# # }
# # lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# rf = RandomForestClassifier(n_jobs=-1)
# # grid_cb = GridSearchCV(cbc, cb_parameters)
# pipe_rf = Pipeline([('rf_boosting', rf)])
# # fit the model
# pipe_rf.fit(X_train, y_train)
# pickle_model(pipe_rf, 'model_pipe_rf')
# print("randomforest stop")
#
# # print("xgboost start")
# # # Initiate and train XGBoostClassifier
# # # cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01],
# # #                     'n_estimators' : [1000, 1500, 2000],
# # #                     'max_depth' : [5, 6, 7, 8, 9 ,10],
# # #                     'num_leaves' : [50, 60, 70, 80, 90, 100]
# # # }
# # # lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# # xgbc = xgb.XGBClassifier()
# # # grid_cb = GridSearchCV(cbc, cb_parameters)
# # pipe_xgbc = Pipeline([('xg_boosting', xgbc)])
# # # fit the model
# # pipe_xgbc.fit(X_train, y_train)
# # pickle_model(pipe_xgbc, 'model_pipe_xgbc')
# # print("xgboost stop")

