from src.data.make_dataset import read_train_data
from src.visualization.visualize import visualise_train_data
from src.features.build_features import build_features
from src.models.train_model import train_models_with_random_search_cv, train_models_with_grid_search_cv
from src.models.train_model_noGridCV import train_no_grid_cv
from src.features.build_features_test import build_features_test_data
from src.data.split_data import split_train_dataset
from src.models.predict_model import evaluate_all_models
from src.models.test_models import test_models_generate_results
import time

t_start = time.time()
"""
Read training data labels and values. Creating training dataset.
"""
df = read_train_data(r'../../data/raw/train_labels.csv', r'../../data/raw/train_values.csv')


"""
Visualise the training data to study features.
Specific to Richter's Predictor: Modeling Earthquake Damage
https://www.drivendata.org/competitions/57/nepal-earthquake
"""
visualise_train_data(df)

"""
Build features for training set.
If dropping all categorical and low corr data use: use_vanilla_data=True (default: False)

To use this routine on a different data set, You have to edit build_features and 
value_column_string in split_train_dataset as
this is specific to Richter's Predictor: Modeling Earthquake Damage
https://www.drivendata.org/competitions/57/nepal-earthquake

use_vanilla_data is made to a global variable since it needs to be consistent 
for train and test datasets
"""
use_vanilla_data = False
train_df = build_features(df, use_vanilla_data=use_vanilla_data)

"""
Split the train dataframe according to value_column_string and train_test_split params
"""
(X_train, X_test, y_train, y_test) = split_train_dataset(train_df,
                                                         value_column_string='damage_grade',
                                                         test_size=0.2,
                                                         random_state=43)

"""
The routine will go through all sklearn classifiers from:
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

and additional:
sklearn.tree.DecisionTree
sklearn.linear_models.SGDClassifier
XGBoost
CatBoost
LGBM
 
Use only baseline model set with use_baseline_models=True (default:False)

Mutually exclusive options for hyperparameter optimization:
Enable GridSearchCV for all models with grid_cv=True (default: False)
Enable RandomSearchCV for all models with random_cv=True (default: False)

GridSearchCV parameter rages done with rule of thumb adequately to a Classifier class
Starting RandomSearchCV were done with a rule of thumb adequately to a Classifier class
Hyperparameter optimization takes a considerable amount of time so use with caution 
"""
# train_models_with_random_search_cv(use_baseline_models=True)
# train_models_with_grid_search_cv(use_baseline_models=True)

# Simplest application
train_no_grid_cv(X_train, y_train, use_baseline_models=True)

"""
Scoring report for all generated models.
The method will test and score the model with F1 micro and macro averaged score
Additionally a cross validation score will be generated for the train dataset
"""
# evaluate_all_models(X_train, y_train, X_test, y_test, cv=3)

"""
Create the test dataset to generate results for upload
"""
test_df = build_features_test_data(r"./../../data/raw/test_values.csv", use_vanilla_data=use_vanilla_data)

"""
Apply test dataset to all trained models ang generate results.
Results in separate files per model found in ../data/results
"""
test_models_generate_results(test_df)

"""
Execute, run away, and pray for the best
"""
t_stop = time.time()
print("The whole procedure took %ss." % (t_stop - t_start))
