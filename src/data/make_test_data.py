import pandas as pd
import os
import category_encoders as ce
from feature_engine.encoding import CountFrequencyEncoder
import matplotlib.pyplot as plt
# from src.features.build_features import dictionary_geo_dam


def create_test_data(test_values_file_path, use_vanilla_data=False):
    dirname = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dirname, test_values_file_path), decimal=",").reset_index(drop=True)

    if use_vanilla_data:
        # Dropping cathegorical from vanilla data
        df = df.drop(['land_surface_condition'], axis=1)
        df = df.drop(['foundation_type'], axis=1)
        df = df.drop(['roof_type'], axis=1)
        df = df.drop(['ground_floor_type'], axis=1)
        df = df.drop(['other_floor_type'], axis=1)
        df = df.drop(['position'], axis=1)
        df = df.drop(['plan_configuration'], axis=1)
        df = df.drop(['legal_ownership_status'], axis=1)

        # Dropping poorly correlated columns
        df = df.drop(['count_families'], axis=1)
        df = df.drop(['has_secondary_use'], axis=1)
        df = df.drop(['has_secondary_use_agriculture'], axis=1)
        df = df.drop(['has_secondary_use_hotel'], axis=1)
        df = df.drop(['has_secondary_use_rental'], axis=1)
        df = df.drop(['has_secondary_use_institution'], axis=1)
        df = df.drop(['has_secondary_use_school'], axis=1)
        df = df.drop(['has_secondary_use_industry'], axis=1)
        df = df.drop(['has_secondary_use_health_post'], axis=1)
        df = df.drop(['has_secondary_use_gov_office'], axis=1)
        df = df.drop(['has_secondary_use_use_police'], axis=1)
        df = df.drop(['has_secondary_use_other'], axis=1)
        test_df = df
        return test_df


    # encoder = ce.binary.BinaryEncoder(cols=None, return_df=True)
    # data = encoder.fit_transform(test_v)
    #
    # #mapping dictionary from build_features
    # data['geo_dam'] = data['geo_level_3_id'].map(dictionary_geo_dam)
    # data['geo_dam'] = data['geo_dam'].fillna(value=0)
    # # print(data['geo_dam'])
    # df = data.copy()
    # # converting geolocation 1 to use a frequency encoder
    # #freq_data_1 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_1_id"])
    # # fitting the encoder
    # #freq_data_1.fit(data_encoded)
    # # creates a dictionary of frequency to categories
    # #dict_1 = freq_data_1.encoder_dict_
    # # creating a new column filled using dictionary
    # #data_encoded["geo_1"] = data_encoded["geo_level_1_id"].map(dict_1['geo_level_1_id'])
    #
    # # converting geolocation 2 to use a frequency encoder
    # #freq_data_2 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_2_id"])
    # # fitting the encoder
    # #freq_data_2.fit(data_encoded)
    # # creates a dictionary of frequency to categories
    # #dict_2 = freq_data_2.encoder_dict_
    # # creating a new column filled using dictionary
    # #data_encoded["geo_2"] = data_encoded["geo_level_2_id"].map(dict_2['geo_level_2_id'])
    #
    # # converting geolocation 3 to use a frequency encoder
    # #freq_data_3 = CountFrequencyEncoder(encoding_method="frequency", variables=["geo_level_3_id"])
    # # fitting the encoder
    # #freq_data_3.fit(data_encoded)
    # # creates a dictionary of frequency to categories
    # #dict_3 = freq_data_3.encoder_dict_
    # # creating a new column filled using dictionary
    # #data_encoded["geo_3"] = data_encoded["geo_level_3_id"].map(dict_3['geo_level_3_id'])
    #
    # # Dropping redundant geolocations
    # #df = data_encoded.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"], axis=1)
    #
    # # Normalising height_percentage, area_percentage, age
    # df['height_p_norm'] = (df['height_percentage'] - df['height_percentage'].min()) / (
    #         df['height_percentage'].max() - df['height_percentage'].min())
    # df['area_p_norm'] = (df['area_percentage'] - df['area_percentage'].min()) / (
    #         df['area_percentage'].max() - df['area_percentage'].min())
    # df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    #
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
    #
    # # Dropping redundant columns
    # test_df = df.drop(["height_percentage", "area_percentage", "age", "geo_level_2_id", "geo_level_3_id", "geo_level_1_id"],axis=1)
    #
    # # remaining_nan = test_df[test_df.isnull().any(axis=1)]
    # # print(remaining_nan)
    #
    # # file_name = '../../data/processed/TestDataSPAM29041745.csv'
    # # df.to_csv(file_name, sep=',')

    test_df = df
    return test_df
