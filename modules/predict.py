import dill
import os
import json

import pandas as pd

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def load_model():
    with open(f'{path}/data/models/{os.listdir(path+"/data/models")[-1]}', 'rb') as file:
        model = dill.load(file)
    return model


def load_test_data():
    data_test_path = path + '/data/test'
    files = os.listdir(data_test_path)
    tests_data = []

    for json_file in files:
        with open(f'{data_test_path}/{json_file}', 'rb') as file:
            tests_data.append(json.load(file))

    df = pd.DataFrame(tests_data)
    return df


def predict():
    model = load_model()
    df_test = load_test_data()
    df_result = pd.DataFrame()

    predicted = model.predict(df_test)
    df_test['result'] = predicted

    df_result[['car_id', 'result']] = df_test[['id', 'result']]
    result_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    with open(result_filename, 'wb') as file:
        df_result.to_csv(file)

if __name__ == '__main__':
    predict()
