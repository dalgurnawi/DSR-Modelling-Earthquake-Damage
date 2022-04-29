import category_encoders as ce
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
import matplotlib.pyplot as plt

# Transforming data to categorical data
from src.data.make_dataset import data
encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
data_encoded = encoder.fit_transform(data)

# converting columns to strings
data_encoded["geo_level_1_id"] = data_encoded["geo_level_1_id"].astype(str)
data_encoded["geo_level_2_id"] = data_encoded["geo_level_2_id"].astype(str)
data_encoded["geo_level_3_id"] = data_encoded["geo_level_3_id"].astype(str)


# converting geolocations to use a frequency encoder
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
# print(print(data_encoded[data_encoded.columns[1:]].corr()['damage_grade'][:]))

print(df.columns)
ov_floors= df.pivot_table(index='damage_grade', columns='count_floors_pre_eq', values='building_id',aggfunc=len, fill_value=0)
print(df["count_floors_pre_eq"].describe())
# ov_floors.plot(kind='bar')
# plt.show()
print(df.loc[:, "count_floors_pre_eq"].value_counts())
# print(df.groupby('count_floors_pre_eq').agg({'damage_grade':'count'}).sort_values(by=['count_floors_pre_eq'], ascending=[True]))
# Determining floor count vs damage grade to see if there are any outliers
# print(df.pivot_table(index="damage_grade", columns="count_floors_pre_eq", values="building_id", aggfunc="count"))
# Dropping instances where floor is above 5
df.drop(df[df.count_floors_pre_eq>5].index, inplace=True)
print(df["count_floors_pre_eq"].describe())

print("Below is a description of the Age column\n")
print(df.age.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
# Majority of data is before age 100. Drop any values above
df.drop(df[df.age>100].index, inplace=True)
print(df["age"].describe())
#normalising age
df['age_norm']=(df['age']-df['age'].min())/(df['age'].max()-df['age'].min())
print(df.area_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
#dropping area percentage over 23
df.drop(df[df.area_percentage>23].index, inplace=True)
# Assessing height_perecentage
# Normalising area_percentage
df['area_p_norm']=(df['area_percentage']-df['area_percentage'].min())/(df['area_percentage'].max()-df['area_percentage'].min())

print(df.height_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
# Dropping any values over 11
df.drop(df[df.height_percentage>11].index, inplace=True)
# Normalising height_percentage
df['height_p_norm']=(df['height_percentage']-df['height_percentage'].min())/(df['height_percentage'].max()-df['height_percentage'].min())

print(df.height_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))

# Reviewing family counts
print(df.count_families.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
print(df.pivot_table(index='damage_grade', columns='count_families', values='building_id',aggfunc=len, fill_value=0))
# Dropping count families over 3
df.drop(df[df.count_families>3].index, inplace=True)

# Dropping redundant columns
df = df.drop(["height_percentage", "area_percentage", "age"],axis=1)
print("Unprocessed data")
print(data_encoded[data_encoded.columns[1:]].corr()['damage_grade'][:])

print("Processed data")

print(df[df.columns[1:]].corr()['damage_grade'][:])
