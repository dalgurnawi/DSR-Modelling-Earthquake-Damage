from sklearn.model_selection import train_test_split
# from src.features.build_features import df
# from src.data.make_dataset import data
# df = data

# # Split the data into 80 to 20 to create a validation set (LEGACY replaced with cross validation score)
# end = int(0.8 * df.shape[0])
# training_set = df.iloc[:end]
# validation_set = df.iloc[end+1:]
# #Prepare data for train_test_split
# X_validation = validation_set.drop(['damage_grade'], axis=1)
# y_validation = validation_set['damage_grade']

# TODO temporary measures for smaller dataset and XGBoost compatibility
# df['damage_grade'] = df['damage_grade'] - 1

# Smaller dataset for testing benchmarks and gridsearches
# df = df.head(10000)

# # Dropping cathegorical from vanilla data
# df = df.drop(['land_surface_condition'], axis=1)
# df = df.drop(['foundation_type'], axis=1)
# df = df.drop(['roof_type'], axis=1)
# df = df.drop(['ground_floor_type'], axis=1)
# df = df.drop(['other_floor_type'], axis=1)
# df = df.drop(['position'], axis=1)
# df = df.drop(['plan_configuration'], axis=1)
# df = df.drop(['legal_ownership_status'], axis=1)
#
# # Dropping poorly correlated columns
# df = df.drop(['count_families'], axis=1)
# df = df.drop(['has_secondary_use'], axis=1)
# df = df.drop(['has_secondary_use_agriculture'], axis=1)
# df = df.drop(['has_secondary_use_hotel'], axis=1)
# df = df.drop(['has_secondary_use_rental'], axis=1)
# df = df.drop(['has_secondary_use_institution'], axis=1)
# df = df.drop(['has_secondary_use_school'], axis=1)
# df = df.drop(['has_secondary_use_industry'], axis=1)
# df = df.drop(['has_secondary_use_health_post'], axis=1)
# df = df.drop(['has_secondary_use_gov_office'], axis=1)
# df = df.drop(['has_secondary_use_use_police'], axis=1)
# df = df.drop(['has_secondary_use_other'], axis=1)

# # dropping columns for testing
# df = df.drop(['land_surface_condition_0'], axis=1)
# df = df.drop(['land_surface_condition_1'], axis=1)
# df = df.drop(['foundation_type_0'], axis=1)
# df = df.drop(['foundation_type_1'], axis=1)
# df = df.drop(['foundation_type_2'], axis=1)
# df = df.drop(['roof_type_0'], axis=1)
# df = df.drop(['roof_type_1'], axis=1)
# df = df.drop(['ground_floor_type_0'], axis=1)
# df = df.drop(['ground_floor_type_1'], axis=1)
# df = df.drop(['ground_floor_type_2'], axis=1)
# df = df.drop(['other_floor_type_0'], axis=1)
# df = df.drop(['other_floor_type_1'], axis=1)
# df = df.drop(['other_floor_type_2'], axis=1)
# df = df.drop(['position_0'], axis=1)
# df = df.drop(['position_1'], axis=1)
# df = df.drop(['position_2'], axis=1)
# df = df.drop(['plan_configuration_0'], axis=1)
# df = df.drop(['plan_configuration_1'], axis=1)
# df = df.drop(['plan_configuration_2'], axis=1)
# df = df.drop(['plan_configuration_3'], axis=1)
# df = df.drop(['legal_ownership_status_0'], axis=1)
# df = df.drop(['legal_ownership_status_1'], axis=1)
# df = df.drop(['legal_ownership_status_2'], axis=1)
# df = df.drop(['count_families'], axis=1)
# df = df.drop(['has_secondary_use'], axis=1)
# df = df.drop(['has_secondary_use_agriculture'], axis=1)
# df = df.drop(['has_secondary_use_hotel'], axis=1)
# df = df.drop(['has_secondary_use_rental'], axis=1)
# df = df.drop(['has_secondary_use_institution'], axis=1)
# df = df.drop(['has_secondary_use_school'], axis=1)
# df = df.drop(['has_secondary_use_industry'], axis=1)
# df = df.drop(['has_secondary_use_health_post'], axis=1)
# df = df.drop(['has_secondary_use_gov_office'], axis=1)
# df = df.drop(['has_secondary_use_use_police'], axis=1)
# df = df.drop(['has_secondary_use_other'], axis=1)


def split_train_dataset(df, value_column_string, test_size=0.2, random_state=43):
    # Prepare data for train_test_split
    y = df[value_column_string].to_numpy()
    X = df.copy()
    X.drop([value_column_string], axis=1, inplace=True)
    X = X.to_numpy()

    # train test split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



