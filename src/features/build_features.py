import category_encoders as ce
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
import matplotlib.pyplot as plt
from src.data.make_dataset import data

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
#print(dictionary)

df = data.copy()

df = df.replace({"damage_grade_str": damage})

df['geo_ref'] = df['damage_grade_str'] + df['geo_level_1_str']

df['geo_dam'] = df['geo_ref'].map(dictionary)

dictionary_geo_dam = {}
for index, row in df.iterrows():
    dictionary_geo_dam[row['geo_level_3_id']] = row['geo_dam']



#for index, row in df.iterrows():
    #df['geo_dam'] = dictionary[row['geo_ref']]

#print(df["count_floors_pre_eq"].describe())

#print(df.loc[:, "count_floors_pre_eq"].value_counts())
# print(df.groupby('count_floors_pre_eq').agg({'damage_grade':'count'}).sort_values(by=['count_floors_pre_eq'], ascending=[True]))
# Determining floor count vs damage grade to see if there are any outliers
# print(df.pivot_table(index="damage_grade", columns="count_floors_pre_eq", values="building_id", aggfunc="count"))

# Dropping instances where floor is above 5
df.drop(df[df.count_floors_pre_eq>5].index, inplace=True)
#print(df["count_floors_pre_eq"].describe())

#print("Below is a description of the Age column\n")
#print(df.age.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
# Majority of data is before age 100. Drop any values above
df.drop(df[df.age>100].index, inplace=True)
#print(df["age"].describe())

#normalising age
df['age_norm']=(df['age']-df['age'].min())/(df['age'].max()-df['age'].min())
#print(df.area_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
#dropping area percentage over 23
df.drop(df[df.area_percentage>23].index, inplace=True)


# Normalising area_percentage
df['area_p_norm']=(df['area_percentage']-df['area_percentage'].min())/(df['area_percentage'].max()-df['area_percentage'].min())

# Assessing height_perecentage
#print(df.height_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
# Dropping any values over 11

df.drop(df[df.height_percentage>11].index, inplace=True)
# Normalising height_percentage
df['height_p_norm']=(df['height_percentage']-df['height_percentage'].min())/(df['height_percentage'].max()-df['height_percentage'].min())

#print(df.height_percentage.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))

# Reviewing family counts
#print(df.count_families.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99]).transpose().reset_index(drop=False))
#print(df.pivot_table(index='damage_grade', columns='count_families', values='building_id',aggfunc=len, fill_value=0))

# Dropping count families over 3
df.drop(df[df.count_families>3].index, inplace=True)

# Dropping redundant columns
df = df.drop(["height_percentage", "area_percentage", "age", "geo_level_2_id", "geo_level_3_id", "geo_level_1_id", "geo_level_1_str", "damage_grade_str", "geo_ref"],axis=1)
#print("Unprocessed data")

#print("Processed data")
#print(df[df.columns[1:]].corr()['damage_grade'][:])

#Adding binary modifiers
encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
df = encoder.fit_transform(df)
# print(df[df.columns[1:]].corr()['damage_grade'][:])

#print(df.columns)

file_name = 'TrainDataSPAMupdated.csv'
df.to_csv(file_name, sep=',')