import os
import pandas as pd
import torch


DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
MODELING_DIR = os.path.join(DATASET_DIR, 'modeling')


def describe_data():
    raw_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_full.csv'))

    print(raw_df.columns)
    print(raw_df.describe())


def construct_train_val_data(columns, window_size=10):
    raw_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_full.csv'))

    for gdf in raw_df.groupby('期权代码'):
        gdf = gdf.sort_values('日期')


if __name__ == '__main__':
    X_labels = ['日期', '行权价', '涨跌幅', '成交额', '收盘价', '结算价', '成交量', '持仓量', 'ETF收盘价']
    describe_data()





