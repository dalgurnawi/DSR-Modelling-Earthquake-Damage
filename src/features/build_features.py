from src.data import make_dataset
from collections import Counter

import category_encoders as ce
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.cluster import KMeans
from feature_engine.encoding import CountFrequencyEncoder


#Transforming data to categorical data
from src.data.make_dataset import data

encoder = ce.binary.BinaryEncoder(cols=None,return_df=True)
#encoder.fit_transform(data.loc[:,"foundation_type"])
# data['found_type_cat'] = enc.fit_transform(data.loc[:,"foundation_type"])
# data['roof_type_cat'] = enc.fit_transform(data.loc[:,"roof_type"])
# data['gf_type_cat'] = enc.fit_transform(data.loc[:,"ground_floor_type"])
# data['of_type_cat'] = enc.fit_transform(data.loc[:,"other_floor_type"])
# data['lsc_category'] = enc.fit_transform(data.loc[:,"land_surface_condition"])
# data['plan_conf_cat'] = enc.fit_transform(data.loc[:,"plan_configuration"])
# data['leg_own_cat'] = enc.fit_transform(data.loc[:,"legal_ownership_status"])
# data['pos_cat'] = enc.fit_transform(data.loc[:,"position"])

data_encoded = encoder.fit_transform(data)

#converting columns to strings
data_encoded["geo_level_1_id"] = data_encoded["geo_level_1_id"].astype(str)
data_encoded["geo_level_2_id"] = data_encoded["geo_level_2_id"].astype(str)
data_encoded["geo_level_3_id"] = data_encoded["geo_level_3_id"].astype(str)


#converting geolocations to use a frequency encoder
freq_data_1 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_1_id"])

#fitting the encoder
freq_data_1.fit(data_encoded)
#creates a dictionary of frequency to categories
dict_1 = freq_data_1.encoder_dict_
#creating a new column filled using dictionary
data_encoded["geo_1"] = data_encoded["geo_level_1_id"].map(dict_1['geo_level_1_id'])

#converting geolocations to use a frequency encoder
freq_data_2 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_2_id"])
#fitting the encoder
freq_data_2.fit(data_encoded)
#creates a dictionary of frequency to categories
dict_2 = freq_data_2.encoder_dict_
#creating a new column filled using dictionary
data_encoded["geo_2"] = data_encoded["geo_level_2_id"].map(dict_2['geo_level_2_id'])

#converting geolocations to use a frequency encoder
freq_data_3 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_3_id"])
#fitting the encoder
freq_data_3.fit(data_encoded)
#creates a dictionary of frequency to categories
dict_3 = freq_data_3.encoder_dict_
#creating a new column filled using dictionary
data_encoded["geo_3"] = data_encoded["geo_level_3_id"].map(dict_3['geo_level_3_id'])


data_encoded = data_encoded.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"],axis=1)
print(print(data_encoded[data_encoded.columns[1:]].corr()['damage_grade'][:]))







