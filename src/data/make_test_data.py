import pandas as pd
import os
import category_encoders as ce
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
import matplotlib.pyplot as plt
from src.data.make_dataset import data

test_v = pd.read_csv(os.path.abspath("./../../data/raw/test_values.csv"), decimal=",").reset_index(drop=True)

encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
data_encoded = encoder.fit_transform(test_v)

# converting columns to strings
data_encoded["geo_level_1_id"] = data_encoded["geo_level_1_id"].astype(str)
data_encoded["geo_level_2_id"] = data_encoded["geo_level_2_id"].astype(str)
data_encoded["geo_level_3_id"] = data_encoded["geo_level_3_id"].astype(str)

freq_data_1 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_1_id"])

# fitting the encoder
freq_data_1.fit(data_encoded)
# creates a dictionary of frequency to categories
dict_1 = freq_data_1.encoder_dict_
# creating a new column filled using dictionary
data_encoded["geo_1"] = data_encoded["geo_level_1_id"].map(dict_1['geo_level_1_id'])

# converting geolocations to use a frequency encoder
freq_data_2 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_2_id"])
# fitting the encoder
freq_data_2.fit(data_encoded)
# creates a dictionary of frequency to categories
dict_2 = freq_data_2.encoder_dict_
# creating a new column filled using dictionary
data_encoded["geo_2"] = data_encoded["geo_level_2_id"].map(dict_2['geo_level_2_id'])

# converting geolocations to use a frequency encoder
freq_data_3 = CountFrequencyEncoder(encoding_method="frequency",variables=["geo_level_3_id"])
# fitting the encoder
freq_data_3.fit(data_encoded)
# creates a dictionary of frequency to categories
dict_3 = freq_data_3.encoder_dict_
# creating a new column filled using dictionary
data_encoded["geo_3"] = data_encoded["geo_level_3_id"].map(dict_3['geo_level_3_id'])

df = data_encoded.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"],axis=1)
df.drop(df[df.count_floors_pre_eq>5].index, inplace=True)
df.drop(df[df.height_percentage>11].index, inplace=True)
# Normalising height_percentage
df['height_p_norm']=(df['height_percentage']-df['height_percentage'].min())/(df['height_percentage'].max()-df['height_percentage'].min())
df.drop(df[df.count_families>3].index, inplace=True)
df = df.drop(["height_percentage", "area_percentage", "age"],axis=1)
file_name = 'TestDataSPAM.csv'
df.to_csv(file_name, sep=',')