from src.data.split_data import X_train, y_train
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import pickle

def pickle_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        f.close()


print("logisticregression start")
# Initiate and train baseline LogisticRegression
lr = LogisticRegression(n_jobs=-1)
# train the initiated model
pipe_lr = Pipeline([('logistic_regression', lr)])
pipe_lr.fit(X_train, y_train)
pickle_model(pipe_lr, 'model_pipe_lr')
print("logisticregression stop")

print("lgbc start")
# Initiate and train LGBMClassifier
# lgbc_parameters = {'learning_rate' : [0.1, 0.05], 'n_estimators' : [1000, 1100], 'max_depth' : [5, 6], 'num_leaves' : [50, 60]}
grid_lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# lgbc = lgb.LGBMClassifier(n_jobs=-1)
# grid_lgbc = GridSearchCV(lgbc, lgbc_parameters, refit=True, verbose=1, n_jobs=-1)
pipe_gb = Pipeline([('gradient_boosting', grid_lgbc)])
# fit the model
pipe_gb.fit(X_train, y_train)
pickle_model(pipe_gb, 'model_pipe_lgbc')
print("lgbc stop")

print("catboost start")
# Initiate and train CatBoostClassifier
# cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01], 'n_estimators' : [1000, 1500, 2000], 'max_depth' : [5, 6, 7, 8, 9 ,10], 'num_leaves' : [50, 60, 70, 80, 90, 100]}
#lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
cbc = cb.CatBoostClassifier()
# grid_cb = GridSearchCV(cbc, cb_parameters)
pipe_cbc = Pipeline([('cat_boosting', cbc)])
# fit the model
pipe_cbc.fit(X_train, y_train)
pickle_model(pipe_cbc, 'model_pipe_cbc')
print("catboost stop")

print("randomforest start")
# Initiate and train CatBoostClassifier
# cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01], 'n_estimators' : [1000, 1500, 2000], 'max_depth' : [5, 6, 7, 8, 9 ,10], 'num_leaves' : [50, 60, 70, 80, 90, 100]}
#lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
rf = RandomForestClassifier(n_jobs=-1)
# grid_cb = GridSearchCV(cbc, cb_parameters)
pipe_rf = Pipeline([('rf_boosting', rf)])
# fit the model
pipe_rf.fit(X_train, y_train)
pickle_model(pipe_rf, 'model_pipe_rf')
print("randomforest stop")

# print("xgboost start")
# # Initiate and train XGBoostClassifier
# # cb_parameters = {'learning_rate' : [0.1, 0.05, 0.01], 'n_estimators' : [1000, 1500, 2000], 'max_depth' : [5, 6, 7, 8, 9 ,10], 'num_leaves' : [50, 60, 70, 80, 90, 100]}
# #lgbc = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, num_leaves=50, n_jobs=-1)
# xgbc = xgb.XGBClassifier()
# # grid_cb = GridSearchCV(cbc, cb_parameters)
# pipe_xgbc = Pipeline([('xg_boosting', xgbc)])
# # fit the model
# pipe_xgbc.fit(X_train, y_train)
# pickle_model(pipe_xgbc, 'model_pipe_xgbc')
# print("xgboost stop")

