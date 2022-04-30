import pandas as pd
import os


def read_train_data(train_data_labels_path, train_data_values_path):
    # Read train data from specified directories
    dirname = os.path.dirname(__file__)
    train_l = pd.read_csv(os.path.join(dirname, train_data_labels_path), decimal=",").reset_index(drop=True)
    train_v = pd.read_csv(os.path.join(dirname, train_data_values_path), decimal=",").reset_index(drop=True)

    # Merging data together using building_id as a reference point
    data = train_v.copy()
    data = data.merge(train_l, how="inner", on="building_id")
    return data

