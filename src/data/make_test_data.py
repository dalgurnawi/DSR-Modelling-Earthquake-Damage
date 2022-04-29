import pandas as pd
import os
import category_encoders as ce
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
import matplotlib.pyplot as plt
from src.data.make_dataset import data
dirname = os.path.dirname(__file__)
test_v = pd.read_csv(os.path.join(dirname, r"./../../data/raw/test_values.csv"), decimal=",").reset_index(drop=True)

encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
data = encoder.fit_transform(test_v)

#convert geo_level locationa and damage_grade to str.
data["geo_level_1_str"] = data["geo_level_1_id"].astype(str)
data["damage_grade_str"] = data["damage_grade"].astype(str)

#Reviewing damage grades per floor count
ov = data.pivot_table(index='geo_level_1_str', columns='damage_grade_str', values='building_id',aggfunc=len, fill_value=0)
#converting overview to df
ov_2 = ov.reset_index()
damage = {'1': 'A', '2':'B', '3':'C'}
ov_2 = ov_2.rename(columns={'1': 'A', '2':'B', '3':'C'})

#total number of samples
n = data.shape[0]

ov_2['A'] = ov_2['A']/n
ov_2['B'] = ov_2['B']/n
ov_2['C'] = ov_2['C']/n

dictionary = {}
for index, row in ov_2.iterrows():
    dictionary['A' + row['geo_level_1_str']] = row['A']
for index, row in ov_2.iterrows():
    dictionary['B' + row['geo_level_1_str']] = row['B']
for index, row in ov_2.iterrows():
    dictionary['C' + row['geo_level_1_str']] = row['C']

df = data.copy()

df = df.replace({"damage_grade_str": damage})

df['geo_ref'] = df['damage_grade_str'] + df['geo_level_1_str']

df['geo_dam'] = df['geo_ref'].map(dictionary)


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
df['height_p_norm'] = (df['height_percentage'] - df['height_percentage'].min()) / (
        df['height_percentage'].max() - df['height_percentage'].min())
df['area_p_norm'] = (df['area_percentage'] - df['area_percentage'].min()) / (
        df['area_percentage'].max() - df['area_percentage'].min())
df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

# Dropping redundant columns
test_df = df.drop(["height_percentage", "area_percentage", "age", "geo_level_2_id", "geo_level_3_id", "geo_level_1_id", "geo_level_1_str", "damage_grade_str", "geo_ref"],axis=1)
file_name = 'TestDataSPAM 29041703.csv'
df.to_csv(file_name, sep=',')
