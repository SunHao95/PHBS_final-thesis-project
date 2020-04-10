__author__ = 'sunhao'

import warnings
import pandas as pd
import numpy as np
import os 
import pickle
import datetime as dt
import statsmodels.api as sm
warnings.filterwarnings("ignore")

trade_date = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../.."))+'\\Data\\raw_data\\wind\\trade_date.csv', index_col=0, dtype={'trade_days':str})
month_date = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../.."))+'\\Data\\raw_data\\wind\\month_date.csv', index_col=0, dtype={'month_date':str})
trade_month_date = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../.."))+'\\Data\\raw_data\\wind\\trade_month_date.csv', index_col=0, dtype={'trade_month_date':str})

def shift_n_trading_day(date, n=1, trade_date=trade_date):
    '''
    获取相对于某一日期移动若干天数的新日期
    param date: "20170821", "xxxxxxxx" (str)
    param n: 1、2、3、-5, 正数表示获取过去的日期，负数表示获取未来的日期（一般不用）

    return：日期 "xxxxxxxx" (str)
    '''
    trade_date = np.array(trade_date)
    assert(type(n) == int)
    if n == 0:
        return date
    if n > 0:
        tds = trade_date[trade_date<date]
    elif n < 0:
        tds = trade_date[trade_date>date]
    tds = tds.tolist()   
    if n > 0:
        return tds[-n]
    elif n < 0:
        n = -n
        return tds[n - 1]

def shift_n_month_day(date, n=1, month_date=trade_month_date):
    '''
    获取相对于某一日期移动若干天数的新日期
    param date: "20170821", "xxxxxxxx" (str)
    param n: 1、2、3、-5, 正数表示获取过去的日期，负数表示获取未来的日期（一般不用）

    return：日期 "xxxxxxxx" (str)
    '''
    month_date = np.array(month_date)
    assert(type(n) == int)
    if n == 0:
        return date
    if n > 0:
        tds = month_date[month_date<date]
    elif n < 0:
        tds = month_date[month_date>date]
    tds = tds.tolist()   
    if n > 0:
        return tds[-n]
    elif n < 0:
        n = -n
        return tds[n - 1]


def pickle_read(file_path):
    fr = open(file_path, 'rb')
    content = pickle.load(fr)
    fr.close()
    return content


def pickle_save(content, file_path):
    fw = open(file_path, 'wb')
    pickle.dump(content, fw, -1)
    fw.close()


def ols_stats(ret, option='nw'):
    if option == 'nw':
        maxlags = int(np.ceil(4 * (len(ret) / 100) ** (2 / 9))) 
        model = sm.OLS(ret.values, np.ones(len(ret))).fit(missing = 'drop', cov_type = 'HAC', cov_kwds={'maxlags':maxlags})
    else:
        model = sm.OLS(ret.values, np.ones(len(ret))).fit()
    return model.tvalues[0], model.pvalues[0]              


def get_rebalance_dates(freq=None, months=None, begt='20100101', endt=None, how='first'):
    '''
    按照指定规则选出调仓日
    param freq: 按天调仓 (int)
    param months: 按月调仓 (list)
    param begt: 调仓起始日 (str)
    param endt: 调仓结束日 (str, 默认到最新交易日的上一个交易日)
    param how: 设置按月调仓时，是月初调仓('first')还是月末调仓('last')

    return: 返回调仓的日期 (list)
    '''
    assert(how in ['first', 'last'])
    if endt is None:
        endt = str(dt.date.today().strftime('%Y%m%d'))
    df = trade_date.copy()
    df = df[(df['trade_days']>=begt) & (df['trade_days']<=endt)]
    
    if freq is not None:
        assert(type(freq) == int)
        return list(df.iloc[::freq, 0].astype(str))
    else:
        default_mons = list(range(1, 13))
        if months == None:
            months = default_mons
        else:
            assert(type(months) == list)
            assert(set(months)<=set(default_mons))
        df.index = pd.DatetimeIndex(df['trade_days'])
        
        df = df['trade_days'].resample(rule='M', how=how)
        return list(df.loc[df.index.month.isin(months)].astype(str))