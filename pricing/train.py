import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
MODELING_DIR = os.path.join(DATASET_DIR, 'modeling')


def describe_data():
    raw_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_full.csv'))

    print(raw_df.columns)
    print(raw_df.describe())


def split_dataset_into_seq(dataset, start_index=0, end_index=None, window_size=6, step=1):
    '''split the dataset to have sequence of observations of length history size'''
    data = []
    start_index = start_index + window_size
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        indices = range(i - window_size, i, step)
        data.append(dataset[indices])
    return np.array(data)


def construct_data_seq(data, data_mean, data_std, window_size=6):
    '''split the dataset into train, val and test splits'''
    # normalization
    data = (data - data_mean) / data_std

    data_in_seq = split_dataset_into_seq(data, start_index=0, end_index=None, window_size=window_size, step=1)

    return data_in_seq


def construct_train_val_data(columns, window_size=10):
    raw_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_full.csv'))

    new_df = pd.DataFrame()
    for code, gdf in raw_df.groupby('期权代码'):
        gdf = gdf.sort_values('日期')
        gdf = gdf.dropna()
        print(code, len(gdf))
        if len(gdf) >= 11:
            new_df = pd.concat([new_df, gdf])

    data_mean = new_df[columns].values.mean(axis=0)
    data_std = new_df[columns].values.std(axis=0)

    data_array = []
    for code, gdf in new_df.groupby('期权代码'):
        gdf = gdf.sort_values('日期')
        gdf = gdf[columns]
        gd = gdf.astype(np.float32).values

        gds = construct_data_seq(gd, data_mean, data_std, window_size=window_size)
        data_array.extend(gds)

    data_array = np.array(data_array)

    return data_array


if __name__ == '__main__':
    features = ['行权价', '涨跌幅', '成交额', '前结算价', '开盘价', '结算价', '成交量', '持仓量',
                'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'ETF收盘价', 'ETF波动率', '国债利率',
                '到期天数', '到期期限', '收盘价']  # r & BS price

    data = construct_train_val_data(features, window_size=11)

    np.save(os.path.join(MODELING_DIR, 'sz50etf_modeling.npy'), data)

    # data = np.load(os.path.join(MODELING_DIR, 'sz50etf_modeling.npy'))






