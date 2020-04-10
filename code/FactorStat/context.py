__author__ = 'sunhao'

import os
import warnings
import pandas as pd
import ps
warnings.filterwarnings("ignore")


class Context(object):
    def __init__(self, begin_year, cut_small=False):
        self.factor_name = None
        self.savedir = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '\\Factor\\'
        self.fee = 0
        self.delist_loss = 0.2
        self.begin_year = begin_year
        self.cut_small = cut_small
        self._cannot_sell = True
        self._factor_reciprocal = False
        self._factor_direction = 'pos'
        self._weighted_method = 'eql_weighted'
        self._ts_data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))+'\\Data\\process_data\\wind\\'
        self._gta_data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))+'\\Data\\raw_data\\gta\\'
        self._get_cannot_buy_dict() 
        self._get_cannot_sell_dict() 
        self._get_delist_stock_dict() 
        self._get_df_factor()
        self._get_df_ret_dict()
        
    def _get_cannot_buy_dict(self):
        begin_date = ps.shift_n_trading_day(str(self.begin_year)+'0101', n=-1)
        self._cannot_buy_dict = ps.pickle_read(self._ts_data_path+'cannot_buy_stock.pkl')
        self._cannot_buy_dict = self._cannot_buy_dict.loc[begin_date:]
    
    def _get_cannot_sell_dict(self):
        begin_date = ps.shift_n_trading_day(str(self.begin_year)+'0131', n=-1)
        self._cannot_sell_dict = ps.pickle_read(self._ts_data_path+'cannot_sell_stock.pkl')
        self._cannot_sell_dict = self._cannot_sell_dict.loc[begin_date:]
        if self.cannot_sell:
            pass
        else:
            for date in self._cannot_sell_dict.index():
                self._cannot_sell_dict[date] = []

    def _get_delist_stock_dict(self):
        begin_date = ps.shift_n_trading_day(str(self.begin_year)+'0131', n=-1)
        self._delist_stock_dict = ps.pickle_read(self._ts_data_path+'delist_stock.pkl')
        self._delist_stock_dict = self._delist_stock_dict.loc[begin_date:]
    
    def _get_df_factor(self):
        begin_date = ps.shift_n_trading_day(str(self.begin_year)+'0101', n=1)
        self._df_factor = ps.pickle_read(self._ts_data_path+'df_factor.pkl')
        self._df_factor = self._df_factor.loc[begin_date:]
        if self.cut_small == True:
            self._df_factor = self._df_factor.groupby(level=0, group_keys=False).apply(lambda x:x.loc[x['circ_mv']>x['circ_mv'].quantile(0.3), :])
        
    def _get_df_ret_dict(self):
        begin_date = ps.shift_n_trading_day(str(self.begin_year)+'0101', n=1)
        self._df_ret_dict = ps.pickle_read(self._ts_data_path+'df_ret.pkl')
        self._df_ret_dict = self._df_ret_dict.loc[begin_date:]


    @property
    def cannot_sell(self):
        return self._cannot_sell

    @cannot_sell.setter
    def cannot_sell(self, value):
        assert value in (True, False)
        self._cannot_sell = value   

    @property
    def factor_reciprocal(self):
        return self._factor_reciprocal

    @factor_reciprocal.setter
    def factor_reciprocal(self, value):
        assert value in (True, False)
        self._factor_reciprocal = value   

    @property
    def factor_direction(self):
        return self._factor_direction

    @factor_direction.setter
    def factor_direction(self, value):
        assert value in ('pos', 'neg')
        self._factor_direction = value   
    
    @property
    def weighted_method(self):
        return self._weighted_method

    @weighted_method.setter
    def weighted_method(self, value):
        assert value in ('eql_weighted', 'mv_weighted')
        self._weighted_method = value
    
    @property
    def cannot_buy_dict(self):
        return self._cannot_buy_dict

    @cannot_buy_dict.setter
    def cannot_buy_dict(self, value):
        self._cannot_buy_dict = value
        
    @property
    def cannot_sell_dict(self):
        return self._cannot_sell_dict

    @cannot_sell_dict.setter
    def cannot_sell_dict(self, value):
        self._cannot_sell_dict = value
        
    @property
    def delist_stock_dict(self):
        return self._delist_stock_dict

    @delist_stock_dict.setter
    def delist_stock_dict(self, value):
        self._delist_stock_dict = value    
        
    @property
    def df_factor(self):
        return self._df_factor

    @df_factor.setter
    def df_factor(self, value):
        self._df_factor = value
        
    @property
    def df_ret_dict(self):
        return self._df_ret_dict

    @df_ret_dict.setter
    def df_ret_dict(self, value):
        self._df_ret_dict = value