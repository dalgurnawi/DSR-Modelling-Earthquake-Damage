import pandas as pd

df_1 = pd.read_csv('../../data/raw/train_values.csv', decimal=',').reset_index(drop=True)
df_2 = pd.read_csv('../../data/raw/train_labels.csv', decimal=',').reset_index(drop=True)
df = df_1.merge(df_2, how='inner', on='building_id')

df = df.drop(['land_surface_condition'], axis=1)
df = df.drop(['foundation_type'], axis=1)
df = df.drop(['roof_type'], axis=1)
df = df.drop(['ground_floor_type'], axis=1)
df = df.drop(['other_floor_type'], axis=1)
df = df.drop(['position'], axis=1)
df = df.drop(['plan_configuration'], axis=1)
df = df.drop(['legal_ownership_status'], axis=1)
