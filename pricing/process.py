import os
import openpyxl
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
# print(DATASET_DIR)


def clean_50etf():
    df = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF.xlsx'))

    df = df.drop([len(df) - 1])
    df = df.drop(columns=['代码', '名称'])
    df = df.set_index(['日期'])

    df = df[(df.index >= '2020-05-01') & (df.index < '2023-05-01')]
    df.to_csv(os.path.join(DATASET_DIR, 'modeling/') + 'sz50etf.csv')
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

    df.to_csv(os.path.join(DATASET_DIR, 'modeling/') + 'sz50etf_option_list.csv')
    print(df)


def clean_50etf_call_option():
    df = pd.DataFrame()
    for sn in range(8, 2, -1):
        sheet = 'Sheet%d' % sn
        pdf = pd.read_excel(os.path.join(DATASET_DIR, '上证50ETF期权.xlsx'), sheet_name=sheet, header=None)

        pdf = pdf.drop([0, 1, 2, 3, 4, 5])
        pdf = pdf.rename(columns=pdf.iloc[0]).iloc[1:]
        pdf = pdf[pdf['交易代码'].str.startswith('510050C')].drop(['期权名称'], axis=1).iloc[::-1]
        pdf['日期'] = pdf['日期'].apply(lambda x: x.date())

        print(pdf)
        df = pd.concat([df, pdf])

    df.to_csv(os.path.join(DATASET_DIR, 'modeling/') + 'sz50etf_option_full.csv', index=False)
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

    df.to_csv(os.path.join(DATASET_DIR, 'modeling/') + 'sz50etf_option_daily_trade.csv', index=False)
    print(df)


if __name__ == '__main__':
    # clean_50etf()
    # clean_50etf_call_option_list()
    # clean_50etf_call_option()
    clean_50etf_call_option_trade()
