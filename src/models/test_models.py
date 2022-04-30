import glob
import os
import pickle
import pandas as pd
import time


def unpickle_model(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model_instance = pickle.load(f)
            f.close()
        return model_instance
    else:
        raise FileNotFoundError


def create_results_for_model(test_df, model_file_name):
    dirname = os.path.dirname(__file__)
    unpickled_model = unpickle_model(model_file_name)
    model_prediction = unpickled_model.predict(test_df)
    results = pd.DataFrame(model_prediction, columns=['damage_grade'])
    upload_data = pd.concat([test_df['building_id'], results['damage_grade']], join='inner', axis=1)
    upload_data = upload_data.rename(columns={'0': 'building_id', '1': 'damage_grade'})
    upload_data.to_csv(os.path.join(dirname, r'../../data/results/%s.csv' % model_file_name.split('/')[-1]), sep=',', index=False)


def test_models_generate_results(test_df):
    t_start = time.time()
    dirname = os.path.dirname(__file__)
    models_list = glob.glob(os.path.join(dirname, r'../models/model*'))
    for model in models_list:
        create_results_for_model(test_df, model)
    t_stop = time.time()
    print(t_stop - t_start)

