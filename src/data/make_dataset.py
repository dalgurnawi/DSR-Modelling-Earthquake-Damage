import pandas as pd
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# File directories'
train_l = pd.read_csv(os.path.abspath("./../../data/raw/train_labels.csv"), decimal=",").reset_index(drop=True)
train_v = pd.read_csv(os.path.abspath("./../../data/raw/train_values.csv"), decimal=",").reset_index(drop=True)

# Merging data together using building_id as a reference point
data = train_v.copy()
data = data.merge(train_l, how="inner", on="building_id")

