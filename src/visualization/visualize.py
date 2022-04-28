import pandas as pd
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
print(train_l.head())
print(train_v.head())


#Merging data together using building_id as a reference point
data = train_v.copy()
data = data.merge(train_l, how="left", on="building_id", right_index=True)
print(data.head())
print(list(data))