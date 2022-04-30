import matplotlib.pyplot as plt
import os


def visualise_train_data(df):
    dirname = os.path.dirname(__file__)
    # Creating an overview of data via pivot table
    ov_floors = df.pivot_table(index='damage_grade', columns='count_floors_pre_eq', values='building_id', aggfunc=len,
                               fill_value=0)
    ov_age = df.pivot_table(index='damage_grade', columns='age', values='building_id', aggfunc=len, fill_value=0)
    ov_area = df.pivot_table(index='damage_grade', columns='area_percentage', values='building_id', aggfunc=len,
                             fill_value=0)
    ov_height = df.pivot_table(index='damage_grade', columns='height_percentage', values='building_id', aggfunc=len,
                               fill_value=0)
    ov_land = df.pivot_table(index='damage_grade', columns='land_surface_condition', values='building_id', aggfunc=len,
                             fill_value=0)
    ov_foundation = df.pivot_table(index='damage_grade', columns='foundation_type', values='building_id', aggfunc=len,
                                   fill_value=0)
    ov_roof = df.pivot_table(index='damage_grade', columns='roof_type', values='building_id', aggfunc=len, fill_value=0)
    ov_gf = df.pivot_table(index='damage_grade', columns='ground_floor_type', values='building_id', aggfunc=len,
                           fill_value=0)
    ov_of = df.pivot_table(index='damage_grade', columns='other_floor_type', values='building_id', aggfunc=len,
                           fill_value=0)
    #
    ov_position = df.pivot_table(index='damage_grade', columns='position', values='building_id', aggfunc=len,
                                 fill_value=0)
    ov_plan = df.pivot_table(index='damage_grade', columns='plan_configuration', values='building_id', aggfunc=len,
                             fill_value=0)
    ov_adobe = df.pivot_table(index='damage_grade', columns='has_superstructure_adobe_mud', values='building_id',
                              aggfunc=len, fill_value=0)
    ov_mud = df.pivot_table(index='damage_grade', columns='has_superstructure_mud_mortar_stone', values='building_id',
                            aggfunc=len, fill_value=0)
    ov_stone = df.pivot_table(index='damage_grade', columns='has_superstructure_stone_flag', values='building_id',
                              aggfunc=len, fill_value=0)
    ov_cement = df.pivot_table(index='damage_grade', columns='has_superstructure_cement_mortar_stone',
                               values='building_id',
                               aggfunc=len, fill_value=0)
    ov_mmb = df.pivot_table(index='damage_grade', columns='has_superstructure_mud_mortar_brick', values='building_id',
                            aggfunc=len, fill_value=0)
    ov_cmb = df.pivot_table(index='damage_grade', columns='has_superstructure_cement_mortar_brick',
                            values='building_id',
                            aggfunc=len, fill_value=0)
    ov_tim = df.pivot_table(index='damage_grade', columns='has_superstructure_timber', values='building_id',
                            aggfunc=len,
                            fill_value=0)
    #
    ov_bam = df.pivot_table(index='damage_grade', columns='has_superstructure_bamboo', values='building_id',
                            aggfunc=len,
                            fill_value=0)
    ov_ne = df.pivot_table(index='damage_grade', columns='has_superstructure_rc_non_engineered', values='building_id',
                           aggfunc=len, fill_value=0)
    ov_eng = df.pivot_table(index='damage_grade', columns='has_superstructure_rc_engineered', values='building_id',
                            aggfunc=len, fill_value=0)
    ov_other = df.pivot_table(index='damage_grade', columns='has_superstructure_other', values='building_id',
                              aggfunc=len,
                              fill_value=0)
    ov_legal = df.pivot_table(index='damage_grade', columns='legal_ownership_status', values='building_id', aggfunc=len,
                              fill_value=0)
    ov_families = df.pivot_table(index='damage_grade', columns='count_families', values='building_id', aggfunc=len,
                                 fill_value=0)
    ov_second = df.pivot_table(index='damage_grade', columns='has_secondary_use', values='building_id', aggfunc=len,
                               fill_value=0)
    ov_agri = df.pivot_table(index='damage_grade', columns='has_secondary_use_agriculture', values='building_id',
                             aggfunc=len, fill_value=0)
    ov_hotel = df.pivot_table(index='damage_grade', columns='has_secondary_use_hotel', values='building_id',
                              aggfunc=len,
                              fill_value=0)

    ov_rent = df.pivot_table(index='damage_grade', columns='has_secondary_use_rental', values='building_id',
                             aggfunc=len,
                             fill_value=0)
    ov_inst = df.pivot_table(index='damage_grade', columns='has_secondary_use_institution', values='building_id',
                             aggfunc=len, fill_value=0)
    ov_school = df.pivot_table(index='damage_grade', columns='has_secondary_use_school', values='building_id',
                               aggfunc=len,
                               fill_value=0)
    ov_industry = df.pivot_table(index='damage_grade', columns='has_secondary_use_industry', values='building_id',
                                 aggfunc=len, fill_value=0)
    ov_health = df.pivot_table(index='damage_grade', columns='has_secondary_use_health_post', values='building_id',
                               aggfunc=len, fill_value=0)
    ov_gov = df.pivot_table(index='damage_grade', columns='has_secondary_use_gov_office', values='building_id',
                            aggfunc=len,
                            fill_value=0)
    ov_pol = df.pivot_table(index='damage_grade', columns='has_secondary_use_use_police', values='building_id',
                            aggfunc=len,
                            fill_value=0)
    ov_su_other = df.pivot_table(index='damage_grade', columns='has_secondary_use_other', values='building_id',
                                 aggfunc=len,
                                 fill_value=0)
    ov_geo = df.pivot_table(index='damage_grade', columns='geo_level_1_id', values='building_id', aggfunc=len,
                            fill_value=0)

    # Using Subplots
    fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    ov_floors.plot(kind='bar', ax=ax[0, 0])
    ov_land.plot(kind='bar', ax=ax[0, 1])
    ov_foundation.plot(kind='bar', ax=ax[0, 2])
    ov_roof.plot(kind='bar', ax=ax[1, 0])
    ov_gf.plot(kind='bar', ax=ax[1, 1])
    ov_of.plot(kind='bar', ax=ax[1, 2])
    plt.savefig(os.path.join(dirname, r"Subplots 1.png"))

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    ov_position.plot(kind='bar', ax=ax[0, 0])
    ov_plan.plot(kind='bar', ax=ax[0, 1])
    ov_adobe.plot(kind='bar', ax=ax[0, 2])
    ov_mud.plot(kind='bar', ax=ax[1, 0])
    ov_stone.plot(kind='bar', ax=ax[1, 1])
    ov_cement.plot(kind='bar', ax=ax[1, 2])
    ov_mmb.plot(kind='bar', ax=ax[2, 0])
    ov_cmb.plot(kind='bar', ax=ax[2, 1])
    ov_tim.plot(kind='bar', ax=ax[2, 2])
    plt.savefig(os.path.join(dirname, r"Subplots 2.png"))

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    ov_bam.plot(kind='bar', ax=ax[0, 0])
    ov_ne.plot(kind='bar', ax=ax[0, 1])
    ov_eng.plot(kind='bar', ax=ax[0, 2])
    ov_other.plot(kind='bar', ax=ax[1, 0])
    ov_legal.plot(kind='bar', ax=ax[1, 1])
    ov_families.plot(kind='bar', ax=ax[1, 2])
    ov_second.plot(kind='bar', ax=ax[2, 0])
    ov_agri.plot(kind='bar', ax=ax[2, 1])
    ov_hotel.plot(kind='bar', ax=ax[2, 2])
    plt.savefig(os.path.join(dirname, r"Subplots 3.png"))
    # fig, ax=plt.subplots(3,3, figsize=(20,20))
    ov_rent.plot(kind='bar', ax=ax[0, 0])
    ov_inst.plot(kind='bar', ax=ax[0, 1])
    ov_school.plot(kind='bar', ax=ax[0, 2])
    ov_industry.plot(kind='bar', ax=ax[1, 0])
    ov_health.plot(kind='bar', ax=ax[1, 1])
    ov_gov.plot(kind='bar', ax=ax[1, 2])
    ov_pol.plot(kind='bar', ax=ax[2, 0])
    ov_su_other.plot(kind='bar', ax=ax[2, 1])
    ov_geo.plot(kind='bar', ax=ax[2, 2])
    plt.savefig(os.path.join(dirname, r"Subplots 4.png"))

    ov_geo.plot(kind='bar', figsize=(20, 20))
    plt.savefig(os.path.join(dirname, r"Subplots 5.png"))
