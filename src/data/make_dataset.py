import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from nancorrmp.nancorrmp import NaNCorrMp
from pandas.testing import assert_frame_equal
import requests
import numpy as np
import seaborn as sn
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats import power
from scipy.stats import shapiro
import warnings
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#File directories'
train_l = pd.read_csv("/Users/davidal-gurnawi/DSR-Modelling-Earthquake-Damage/data/raw/train_labels.csv", decimal=",").reset_index(drop=True)
train_v = pd.read_csv("/Users/davidal-gurnawi/DSR-Modelling-Earthquake-Damage/data/raw/train_values.csv", decimal=",").reset_index(drop=True)


#Viewing data file headers
# print(train_l.head())
# print(train_v.head())


# print(train_l.dtypes)
# print(train_v.dtypes)


#Merging data together using building_id as a reference point
data = train_v.copy()
data = data.merge(train_l, how="inner", on="building_id", right_index=True)
# print(data.head())
# print(list(data))
#
# #Viewing data description
# print(data.describe())
#
# #Plotting histograms of data
# data.hist(bins=100, figsize=(20,15))
#
# #Correlation of target variable against features
# print(data[data.columns[1:]].corr()['damage_grade'][:])
#
# #determining if there are any null values
# print(data.loc[:,"foundation_type"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"roof_type"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"ground_floor_type"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"other_floor_type"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"land_surface_condition"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"plan_configuration"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"legal_ownership_status"].value_counts().sort_index(ascending=False))
# print(data.loc[:,"position"].value_counts().sort_index(ascending=False))
