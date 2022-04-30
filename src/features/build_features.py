import category_encoders as ce
from feature_engine.encoding import CountFrequencyEncoder


def build_features(data, use_vanilla_data=False):
    df = data

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

    # Dropping outliers
    df.drop(df[df.count_floors_pre_eq > 5].index, inplace=True)
    df.drop(df[df.age > 100].index, inplace=True)
    df.drop(df[df.area_percentage > 23].index, inplace=True)
    df.drop(df[df.height_percentage > 11].index, inplace=True)
    df.drop(df[df.count_families > 3].index, inplace=True)

    # Normalising continuous variables
    df['height_p_norm'] = (df['height_percentage'] - df['height_percentage'].min()) / (
            df['height_percentage'].max() - df['height_percentage'].min())
    df['area_p_norm'] = (df['area_percentage'] - df['area_percentage'].min()) / (
            df['area_percentage'].max() - df['area_percentage'].min())
    df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

    # dropping redundant variables
    df.drop(
        ['height_percentage', 'area_percentage', 'age', 'geo_level_2_id', 'geo_level_3_id',
         'has_secondary_use_agriculture',
         'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
         'has_secondary_use_school', 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
         'has_secondary_use_use_police', 'has_secondary_use_other'], axis=1, inplace=True)

    # Converting features to str to get encoder to work
    df['count_floors_pre_eq'] = df['count_floors_pre_eq'].astype(str)
    df['geo_level_1_id'] = df['geo_level_1_id'].astype(str)
    df['count_families'] = df['count_families'].astype(str)

    # set up the encoder
    encoder = CountFrequencyEncoder(encoding_method='frequency',
                                    variables=['geo_level_1_id', 'count_floors_pre_eq', 'land_surface_condition',
                                               'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
                                               'position', 'plan_configuration', 'legal_ownership_status',
                                               'count_families'])
    # fit the encoder
    encoder.fit(df)
    df = encoder.transform(df)

    # Adding binary modifiers
    binary = ce.binary.BinaryEncoder(
        cols=['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
              'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick',
              'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo',
              'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other',
              'has_secondary_use', 'has_secondary_use_hotel', 'has_secondary_use_rental'], return_df=True)
    train_df = binary.fit_transform(df)

    return train_df
