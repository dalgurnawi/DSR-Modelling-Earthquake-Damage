import pandas as pd
import category_encoders as ce
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
from feature_engine.encoding import CountFrequencyEncoder
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os

dirname = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dirname, r"./../../data/raw/test_values.csv"), decimal=",").reset_index(drop=True)

# encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
# data = encoder.fit_transform(test_v)

#mapping dictionary from build_features
# data['geo_dam'] = data['geo_level_3_id'].map(dictionary_geo_dam)
# data['geo_dam'] = data['geo_dam'].fillna(value=0)
# print(data['geo_dam'])
# df = data.copy()
# converting geolocation 1 to use a frequency encoder
#freq_data_1 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_1_id"])
# fitting the encoder
#freq_data_1.fit(data_encoded)
# creates a dictionary of frequency to categories
#dict_1 = freq_data_1.encoder_dict_
# creating a new column filled using dictionary
#data_encoded["geo_1"] = data_encoded["geo_level_1_id"].map(dict_1['geo_level_1_id'])

# converting geolocation 2 to use a frequency encoder
#freq_data_2 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_2_id"])
# fitting the encoder
#freq_data_2.fit(data_encoded)
# creates a dictionary of frequency to categories
#dict_2 = freq_data_2.encoder_dict_
# creating a new column filled using dictionary
#data_encoded["geo_2"] = data_encoded["geo_level_2_id"].map(dict_2['geo_level_2_id'])

# converting geolocation 3 to use a frequency encoder
#freq_data_3 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_3_id"])
# fitting the encoder
#freq_data_3.fit(data_encoded)
# creates a dictionary of frequency to categories
#dict_3 = freq_data_3.encoder_dict_
# creating a new column filled using dictionary
#data_encoded["geo_3"] = data_encoded["geo_level_3_id"].map(dict_3['geo_level_3_id'])

# Dropping redundant geolocations
#df = data_encoded.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"], axis=1)

# Normalising height_percentage, area_percentage, age
# df['height_p_norm'] = (df['height_percentage'] - df['height_percentage'].min()) / (
#         df['height_percentage'].max() - df['height_percentage'].min())
# df['area_p_norm'] = (df['area_percentage'] - df['area_percentage'].min()) / (
#         df['area_percentage'].max() - df['area_percentage'].min())
# df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

# Dropping redundant columns
# test_df = df.drop(["height_percentage", "area_percentage", "age", "geo_level_2_id", "geo_level_3_id", "geo_level_1_id"],axis=1)

# remaining_nan = test_df[test_df.isnull().any(axis=1)]
# print(remaining_nan)
#
# file_name = 'TestDataSPAM29041745.csv'
# df.to_csv(file_name, sep=',')

############################################################################################
#Normalising continuous variables
df['height_p_norm']=(df['height_percentage']-df['height_percentage'].min())/(df['height_percentage'].max()-df['height_percentage'].min())
df['area_p_norm']=(df['area_percentage']-df['area_percentage'].min())/(df['area_percentage'].max()-df['area_percentage'].min())
df['age_norm']=(df['age']-df['age'].min())/(df['age'].max()-df['age'].min())

#dropping redundant variables
df.drop(['height_percentage','area_percentage','age', 'geo_level_2_id','geo_level_3_id','has_secondary_use_agriculture','has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry','has_secondary_use_school', 'has_secondary_use_health_post', 'has_secondary_use_gov_office','has_secondary_use_use_police', 'has_secondary_use_other'], axis=1, inplace=True)

#Converting features to str to get encoder to work
df['count_floors_pre_eq'] = df['count_floors_pre_eq'].astype(str)
df['geo_level_1_id'] = df['geo_level_1_id'].astype(str)
df['count_families'] = df['count_families'].astype(str)

# set up the encoder
encoder = CountFrequencyEncoder(encoding_method='frequency', variables=['geo_level_1_id', 'count_floors_pre_eq','land_surface_condition','foundation_type', 'roof_type','ground_floor_type','other_floor_type','position','plan_configuration', 'legal_ownership_status', 'count_families'])
# fit the encoder
encoder.fit(df)
df = encoder.transform(df)

#Adding binary modifiers
binary = ce.binary.BinaryEncoder(cols=['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone','has_superstructure_stone_flag','has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use','has_secondary_use_hotel','has_secondary_use_rental'], return_df=True)
df = binary.fit_transform(df)
print(df.head())