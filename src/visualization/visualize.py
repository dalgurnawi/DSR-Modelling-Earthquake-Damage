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
train_l = pd.read_csv('')
train_v = '/Users/davidal-gurnawi/DSR-Modelling-Earthquake-Damage/data/raw/train_values.csv'

#Import
