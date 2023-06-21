import pandas as pd
import requests
from csv import reader
import os

URL = 'http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag=Validation%2FmAP&run=crop_True_flip_True_solarize_True_gauss_False_sgd_gridv3_sf_32_negr2.0_nsm_0.3_lgminiou_0.5_nodes_4800_lr_0.0001_bs_16_2023-06-20_22-13-38&format=csv'

def URLs(run_name):
    URLs_dict = {
    'val_accuracy' : f'http://localhost:6006/data/plugin/scalars/scalars?tag=Validation%2FmAP&run={run_name}&format=csv',
    }
    return URLs_dict

def tb_data(log_dir, mode, num_trials):
    run_names = os.listdir(log_dir)
    fdf = {}
    for i, run_name in enumerate(run_names[:num_trials]):
        r = requests.get(URLs(run_name)[mode], allow_redirects=True)
        data = r.text
        data_csv = reader(data.splitlines())
        data_csv = list(data_csv)
        df = pd.DataFrame(data_csv)
        headers = df.iloc[0]
        df  = pd.DataFrame(df.values[1:], columns=headers)
        if i == 0:
            fdf['Step'] = df['Step']  
        fdf[f'trial {trial}'] = df['Value']
    fdf = pd.DataFrame(fdf)
    return fdf


def main():
    log_dir = './runs/runs'
    num_trials = len(os.listdir(log_dir))
    mode = 'val_accuracy'
    df = tb_data(log_dir, mode, num_trials)
    df.to_csv(f'{mode}.csv', index=False)


if __name__ == '__main__':
    main()
