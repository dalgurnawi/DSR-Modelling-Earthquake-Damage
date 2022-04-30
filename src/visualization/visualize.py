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
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# File directories'
dirname = os.path.dirname(__file__)
train_l = pd.read_csv(os.path.join(dirname, r"./../../data/raw/train_labels.csv"), decimal=",").reset_index(drop=True)
train_v = pd.read_csv(os.path.join(dirname, r"./../../data/raw/train_values.csv"), decimal=",").reset_index(drop=True)


# Viewing data file headers
#print(train_l.head())
#print(train_v.head())
# Merging data together using building_id as a reference point
df = train_v.copy()
df = df.merge(train_l, how="inner", on="building_id")
#print(df.head())
#print(list(df))
#print(df.head())

# Creating an overview of data via pivot table
ov_floors= df.pivot_table(index='damage_grade', columns='count_floors_pre_eq', values='building_id',aggfunc=len, fill_value=0)
ov_age=df.pivot_table(index='damage_grade', columns='age', values='building_id',aggfunc=len, fill_value=0)
ov_area=df.pivot_table(index='damage_grade', columns='area_percentage', values='building_id',aggfunc=len, fill_value=0)
ov_height=df.pivot_table(index='damage_grade', columns='height_percentage', values='building_id',aggfunc=len, fill_value=0)
ov_land=df.pivot_table(index='damage_grade', columns='land_surface_condition', values='building_id',aggfunc=len, fill_value=0)
ov_foundation=df.pivot_table(index='damage_grade', columns='foundation_type', values='building_id',aggfunc=len, fill_value=0)
ov_roof=df.pivot_table(index='damage_grade', columns='roof_type', values='building_id',aggfunc=len, fill_value=0)
ov_gf=df.pivot_table(index='damage_grade', columns='ground_floor_type', values='building_id',aggfunc=len, fill_value=0)
ov_of=df.pivot_table(index='damage_grade', columns='other_floor_type', values='building_id',aggfunc=len, fill_value=0)
#
ov_position=df.pivot_table(index='damage_grade', columns='position', values='building_id',aggfunc=len, fill_value=0)
ov_plan=df.pivot_table(index='damage_grade', columns='plan_configuration', values='building_id',aggfunc=len, fill_value=0)
ov_adobe=df.pivot_table(index='damage_grade', columns='has_superstructure_adobe_mud', values='building_id',aggfunc=len, fill_value=0)
ov_mud=df.pivot_table(index='damage_grade', columns='has_superstructure_mud_mortar_stone', values='building_id',aggfunc=len, fill_value=0)
ov_stone=df.pivot_table(index='damage_grade', columns='has_superstructure_stone_flag', values='building_id',aggfunc=len, fill_value=0)
ov_cement=df.pivot_table(index='damage_grade', columns='has_superstructure_cement_mortar_stone', values='building_id',aggfunc=len, fill_value=0)
ov_mmb=df.pivot_table(index='damage_grade', columns='has_superstructure_mud_mortar_brick', values='building_id',aggfunc=len, fill_value=0)
ov_cmb=df.pivot_table(index='damage_grade', columns='has_superstructure_cement_mortar_brick', values='building_id',aggfunc=len, fill_value=0)
ov_tim=df.pivot_table(index='damage_grade', columns='has_superstructure_timber', values='building_id',aggfunc=len, fill_value=0)
#
ov_bam=df.pivot_table(index='damage_grade', columns='has_superstructure_bamboo', values='building_id',aggfunc=len, fill_value=0)
ov_ne=df.pivot_table(index='damage_grade', columns='has_superstructure_rc_non_engineered', values='building_id',aggfunc=len, fill_value=0)
ov_eng=df.pivot_table(index='damage_grade', columns='has_superstructure_rc_engineered', values='building_id',aggfunc=len, fill_value=0)
ov_other=df.pivot_table(index='damage_grade', columns='has_superstructure_other', values='building_id',aggfunc=len, fill_value=0)
ov_legal=df.pivot_table(index='damage_grade', columns='legal_ownership_status', values='building_id',aggfunc=len, fill_value=0)
ov_families=df.pivot_table(index='damage_grade', columns='count_families', values='building_id',aggfunc=len, fill_value=0)
ov_second=df.pivot_table(index='damage_grade', columns='has_secondary_use', values='building_id',aggfunc=len, fill_value=0)
ov_agri=df.pivot_table(index='damage_grade', columns='has_secondary_use_agriculture', values='building_id',aggfunc=len, fill_value=0)
ov_hotel=df.pivot_table(index='damage_grade', columns='has_secondary_use_hotel', values='building_id',aggfunc=len, fill_value=0)
#
ov_rent=df.pivot_table(index='damage_grade', columns='has_secondary_use_rental', values='building_id',aggfunc=len, fill_value=0)
ov_inst=df.pivot_table(index='damage_grade', columns='has_secondary_use_institution', values='building_id',aggfunc=len, fill_value=0)
ov_school=df.pivot_table(index='damage_grade', columns='has_secondary_use_school', values='building_id',aggfunc=len, fill_value=0)
ov_industry=df.pivot_table(index='damage_grade', columns='has_secondary_use_industry', values='building_id',aggfunc=len, fill_value=0)
ov_health=df.pivot_table(index='damage_grade', columns='has_secondary_use_health_post', values='building_id',aggfunc=len, fill_value=0)
ov_gov=df.pivot_table(index='damage_grade', columns='has_secondary_use_gov_office', values='building_id',aggfunc=len, fill_value=0)
ov_pol=df.pivot_table(index='damage_grade', columns='has_secondary_use_use_police', values='building_id',aggfunc=len, fill_value=0)
ov_su_other=df.pivot_table(index='damage_grade', columns='has_secondary_use_other', values='building_id',aggfunc=len, fill_value=0)
ov_geo=df.pivot_table(index='damage_grade', columns='geo_level_1_id', values='building_id',aggfunc=len, fill_value=0)


#Using Subplots
fig, ax=plt.subplots(2,3, figsize=(20,20))
ov_floors.plot(kind='bar', ax=ax[0,0])
ov_land.plot(kind='bar', ax=ax[0,1])
ov_foundation.plot(kind='bar', ax=ax[0,2])
ov_roof.plot(kind='bar', ax=ax[1,0])
ov_gf.plot(kind='bar', ax=ax[1,1])
ov_of.plot(kind='bar', ax=ax[1,2])
plt.savefig("Subplots 1.png")
# #
fig, ax=plt.subplots(3,3, figsize=(20,20))
ov_position.plot(kind='bar', ax=ax[0,0])
ov_plan.plot(kind='bar', ax=ax[0,1])
ov_adobe.plot(kind='bar', ax=ax[0,2])
ov_mud.plot(kind='bar', ax=ax[1,0])
ov_stone.plot(kind='bar', ax=ax[1,1])
ov_cement.plot(kind='bar', ax=ax[1,2])
ov_mmb.plot(kind='bar', ax=ax[2,0])
ov_cmb.plot(kind='bar', ax=ax[2,1])
ov_tim.plot(kind='bar', ax=ax[2,2])
plt.savefig("Subplots 2.png")
fig, ax=plt.subplots(3,3, figsize=(20,20))
ov_bam.plot(kind='bar', ax=ax[0,0])
ov_ne.plot(kind='bar', ax=ax[0,1])
ov_eng.plot(kind='bar', ax=ax[0,2])
ov_other.plot(kind='bar', ax=ax[1,0])
ov_legal.plot(kind='bar', ax=ax[1,1])
ov_families.plot(kind='bar', ax=ax[1,2])
ov_second.plot(kind='bar', ax=ax[2,0])
ov_agri.plot(kind='bar', ax=ax[2,1])
ov_hotel.plot(kind='bar', ax=ax[2,2])
plt.savefig("Subplots 3.png")
# fig, ax=plt.subplots(3,3, figsize=(20,20))
ov_rent.plot(kind='bar', ax=ax[0,0])
ov_inst.plot(kind='bar', ax=ax[0,1])
ov_school.plot(kind='bar', ax=ax[0,2])
ov_industry.plot(kind='bar', ax=ax[1,0])
ov_health.plot(kind='bar', ax=ax[1,1])
ov_gov.plot(kind='bar', ax=ax[1,2])
ov_pol.plot(kind='bar', ax=ax[2,0])
ov_su_other.plot(kind='bar', ax=ax[2,1])
ov_geo.plot(kind='bar', ax=ax[2,2])
plt.savefig("Subplots 4.png")

ov_geo.plot(kind='bar', figsize=(20,20))
plt.savefig("Subplots 5.png")
# Dropping some variables
#print("correlation prior to variable dropping")
#print(print(df[df.columns[1:]].corr()['damage_grade'][:]))

#Dropping outliers
df.drop(df[df.count_floors_pre_eq>5].index, inplace=True)
df.drop(df[df.age>100].index, inplace=True)
df.drop(df[df.area_percentage>23].index, inplace=True)
df.drop(df[df.height_percentage>11].index, inplace=True)
df.drop(df[df.count_families>3].index, inplace=True)
# print("correlation after feature dropping")
# print(df[df.columns[1:]].corr()['damage_grade'][:])

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
print(df)