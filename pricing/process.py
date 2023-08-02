import os
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
MODELING_DIR = os.path.join(DATASET_DIR, 'modeling')
# print(DATASET_DIR)

TRADE_DAYS = 244


def clean_50etf():
    df = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF.xlsx'))

    df = df.drop([len(df) - 1])
    df = df.drop(columns=['代码', '名称'])
    df = df.set_index(['日期'])

    df = df[(df.index >= '2020-03-01') & (df.index < '2023-05-01')]

    # 波动率
    df['波动率'] = df['收盘价'].pct_change().rolling(30).std() * np.sqrt(TRADE_DAYS)

    df = df[(df.index >= '2020-05-01') & (df.index < '2023-05-01')]

    df.to_csv(os.path.join(MODELING_DIR, 'sz50etf.csv'))
    print(df)


def clean_50etf_call_option_list():
    df = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF期权.xlsx'), header=None)

    df = df.drop([0, 1, 2, 3, 4])
    df = df.rename(columns=df.iloc[0]).iloc[1:]
    df = df[df['认购认沽'] == '认购']
    df = df[['期权代码', '交易代码', '行权价', '合约单位', '期权上市日', '期权到期日', '上市日开盘参考价']]
    df = df.set_index(['期权代码'])
    df['期权上市日'] = df['期权上市日'].apply(lambda x: x.date())
    df['期权到期日'] = df['期权到期日'].apply(lambda x: x.date())

    df.to_csv(os.path.join(MODELING_DIR, 'sz50etf_option_list.csv'))
    print(df)


def clean_50etf_call_option_quotation():
    df = pd.DataFrame()
    for sn in range(8, 2, -1):
        sheet = 'Sheet%d' % sn
        pdf = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF期权.xlsx'), sheet_name=sheet, header=None)

        pdf = pdf.drop([0, 1, 2, 3, 4, 5])
        pdf = pdf.rename(columns=pdf.iloc[0]).iloc[1:]
        pdf = pdf[pdf['交易代码'].str.startswith('510050C')].drop(['期权名称'], axis=1).iloc[::-1]
        pdf['日期'] = pdf['日期'].apply(lambda x: x.date())
        pdf.sort_values(['日期', '期权代码'], ascending=[True, True], inplace=True)

        print(pdf)
        df = pd.concat([df, pdf])

    df.to_csv(os.path.join(MODELING_DIR, 'sz50etf_option_quotation.csv'), index=False)
    print(df)


def clean_50etf_call_option_trade():
    df = pd.DataFrame()
    for sn in range(12, 8, -1):
        sheet = 'Sheet%d' % sn
        pdf = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF期权.xlsx'), sheet_name=sheet, header=None)

        pdf = pdf.drop([0, 1, 2, 3, 4, 5])
        pdf = pdf.rename(columns=pdf.iloc[0]).iloc[1:]
        pdf = pdf.drop(['标的代码', '标的名称'], axis=1).iloc[::-1]
        pdf['日期'] = pdf['日期'].apply(lambda x: x.date())

        print(pdf)
        df = pd.concat([df, pdf])

    df.to_csv(os.path.join(MODELING_DIR, 'sz50etf_option_daily_trade.csv'), index=False)
    print(df)


def clean_50etf_call_option_full():
    asset_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf.csv'))
    option_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_list.csv'))
    quotation_df = pd.read_csv(os.path.join(MODELING_DIR, 'sz50etf_option_quotation.csv'))
    treasure_df = pd.read_csv(os.path.join(MODELING_DIR, 'treasure_bond_1y_yield.csv'))

    asset_df = asset_df.rename(columns={'收盘价': 'ETF收盘价', '波动率': 'ETF波动率'})
    # select option list after 2020-05-01 and delist before 2023-05-01
    option_df = option_df[(option_df['期权上市日'] >= '2020-05-01') & (option_df['期权到期日'] <= '2023-05-01')]
    df = quotation_df[quotation_df['期权代码'].isin(option_df['期权代码'])]

    df = df.merge(asset_df[['日期', 'ETF收盘价', 'ETF波动率']], on='日期', how='left')
    # df['ETF收盘价'] = df['日期'].apply(lambda x: asset_df[asset_df['日期'] == x].iloc[0]['收盘价'])
    # df['ETF波动率'] = df['日期'].apply(lambda x: asset_df[asset_df['日期'] == x].iloc[0]['波动率'])
    df = df.merge(option_df[['期权代码', '合约单位', '期权上市日', '期权到期日']], on='期权代码', how='left')
    # df['合约单位'] = df['期权代码'].apply(lambda x: option_df[option_df['期权代码'] == x].iloc[0]['合约单位'])
    # df['期权上市日'] = df['期权代码'].apply(lambda x: option_df[option_df['期权代码'] == x].iloc[0]['期权上市日'])
    # df['期权到期日'] = df['期权代码'].apply(lambda x: option_df[option_df['期权代码'] == x].iloc[0]['期权到期日'])

    df['日期'] = pd.to_datetime(df['日期'])
    df['期权到期日'] = pd.to_datetime(df['期权到期日'])

    treasure_df['日期'] = pd.to_datetime(treasure_df['日期'])
    treasure_df = treasure_df.rename(columns={'收盘': '国债利率'})
    # 国债利率
    df = df.merge(treasure_df[['日期', '国债利率']], on='日期', how='left')
    # df['国债利率'] = df['日期'].apply(lambda x: treasure_df[treasure_df['日期'] == x].iloc[0]['收盘'])

    df = df.fillna(method='ffill')

    df['到期天数'] = df['期权到期日'] - df['日期']
    df['到期天数'] = (df['到期天数'].astype(int) / 1000000000 / 60 / 60 / 24).astype(int)
    df['到期期限'] = df['到期天数'] / 365

    df.to_csv(os.path.join(MODELING_DIR, 'sz50etf_option_full.csv'), index=False)
    print(df)


if __name__ == '__main__':
    # clean_50etf()
    # clean_50etf_call_option_list()
    # clean_50etf_call_option_quotation()
    # clean_50etf_call_option_trade()
    clean_50etf_call_option_full()

    pass


# https://finance.sina.com.cn/money/future/roll/2019-12-22/doc-iihnzhfz7598803.shtml
# https://finance.sina.com.cn/money/future/roll/2019-12-22/doc-iihnzhfz7598803.shtml


