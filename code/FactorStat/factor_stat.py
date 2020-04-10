__author__ = 'sunhao'

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import ps

warnings.filterwarnings("ignore")

class FactorStat(object):

    def __init__(self, context):
        self.context = context
        self._get_pos_hold()
        self._get_group_ret()
        self._get_month_ret()
        self._stat_analysis()
        
    def _cal_factor_pos(self, date):
        df = self.context.df_factor.loc[date][['circ_mv', self.context.factor_name]].dropna(axis=0, how='any')
        if self.context.factor_reciprocal:
            df[self.context.factor_name] = 1 / df[self.context.factor_name]
        df['mv_cut'] = pd.qcut(df['circ_mv'], 2, ['small', 'large'])
        df['factor_cut'] = None
        df.loc[df['mv_cut']  == 'small', 'factor_cut'] = pd.qcut(df.loc[df['mv_cut']  == 'small', self.context.factor_name], [0, 0.3, 0.7, 1], ['low', 'medium', 'high'])
        df.loc[df['mv_cut']  == 'large', 'factor_cut'] = pd.qcut(df.loc[df['mv_cut']  == 'large', self.context.factor_name], [0, 0.3, 0.7, 1], ['low', 'medium', 'high'])
        buy_date = ps.shift_n_trading_day(date, n=-1)
        df = df.loc[list(set(df.index.tolist()) - set(self.context.cannot_buy_dict[buy_date]))]
        df_large_high = df.query('mv_cut=="large" and factor_cut=="high"')
        df_small_high = df.query('mv_cut=="small" and factor_cut=="high"')
        df_large_low = df.query('mv_cut=="large" and factor_cut=="low"')
        df_small_low = df.query('mv_cut=="small" and factor_cut=="low"')

        if self.context.weighted_method == 'eql_weighted':
            def cal_weight(df):
                return pd.Series(index = df.index, name='eql_weighted').fillna(1/len(df))
            return cal_weight(df_large_high), cal_weight(df_small_high), cal_weight(df_large_low), cal_weight(df_small_low)
    
        if self.context.weighted_method == 'mv_weighted':
            def cal_weight(df):
                return df['circ_mv'] / df['circ_mv'].sum()
            return cal_weight(df_large_high), cal_weight(df_small_high), cal_weight(df_large_low), cal_weight(df_small_low)

    def _get_pos_hold(self):
        self.large_high_pos = {}
        self.small_high_pos = {}
        self.large_low_pos = {}
        self.small_low_pos = {}
        for month_date in sorted(list(self.context.df_ret_dict.keys())):
            buy_date = ps.shift_n_trading_day(month_date, -1)
            self.large_high_pos[buy_date], self.small_high_pos[buy_date], self.large_low_pos[buy_date], self.small_low_pos[buy_date] =\
            self._cal_factor_pos(month_date)
        
    def _cal_ret(self, pos_hold_raw):
        ret_list = []
        turnover_dict = {}
        delist_turnover_dict = {}
        pos_hold = pos_hold_raw.copy()
        rebalance_dates = sorted(list(pos_hold.keys()))
        for i, date in enumerate(rebalance_dates):
            pre_date = ps.shift_n_trading_day(date)
            df = pd.merge(pos_hold[date].to_frame(), self.context.df_ret_dict[pre_date].iloc[:, 1:], left_index=True, right_index=True, how='left').fillna(0)
            ret = df.iloc[:, 1:].apply(np.average, weights=df.iloc[:, 0])
            if i == 0:
                ret = ret.append(pd.Series(index=[date], data=0))
                turnover_dict[date] = 1.0
            if i < len(rebalance_dates) - 1:
                nav_end = df.iloc[:, 1:].multiply(df.iloc[:, 0], axis=0).add(1).prod(axis=1)
                wgt_end = nav_end / nav_end.sum()
                cannot_sell_stock = self.context.cannot_sell_dict[df.iloc[:, -1].name]
                delist_stock = self.context.delist_stock_dict[df.iloc[:, -1].name]
                delist_stock = set(cannot_sell_stock).intersection(set(delist_stock))
                if len(delist_stock)!=0:
                    cannot_sell_stock = set(cannot_sell_stock) - set(delist_stock)
                    delist_turnover = wgt_end.loc[wgt_end.index.isin(delist_stock)].sum() 
                else:
                    delist_turnover = 0 
                delist_turnover_dict[date] = delist_turnover
                wgt_cannot_sell = wgt_end.loc[wgt_end.index.isin(cannot_sell_stock)]
                origin_pos = pos_hold[df.iloc[:, -1].name]
                new_pos = (1 - wgt_cannot_sell.sum()) * origin_pos
                new_pos = new_pos.append(wgt_cannot_sell).groupby(level=0).sum()
                pos_hold[df.iloc[:, -1].name] = new_pos
                turnover_rate = pd.merge(wgt_end.to_frame(), new_pos.to_frame(), left_index=True, right_index=True, how='outer').fillna(0).diff(axis=1).abs().sum().iloc[-1]
                ret_list.append(ret) 
                turnover_dict[df.iloc[:, -1].name] = turnover_rate
            else:
                ret_list.append(ret)
        res = pd.concat(ret_list, axis=0).sort_index(ascending=True)
        df_turnover = pd.Series(turnover_dict)
        df_delist_turnover = pd.Series(delist_turnover_dict)
        return res, df_turnover, df_delist_turnover
    
    def _get_group_ret(self):
        self._large_high_ret, self._large_high_turnover, self._large_high_delist_turnover = self._cal_ret(self.large_high_pos)
        self._small_high_ret, self._small_high_turnover, self._small_high_delist_turnover = self._cal_ret(self.small_high_pos)
        self._large_low_ret, self._large_low_turnover, self._large_low_delist_turnover = self._cal_ret(self.large_low_pos)
        self._small_low_ret, self._small_low_turnover, self._small_low_delist_turnover = self._cal_ret(self.small_low_pos)
        
        high_fee = (self._large_high_turnover.reindex(self._large_high_ret.index).fillna(0) + self._small_high_turnover.reindex(self._small_high_ret.index).fillna(0)) * self.context.fee +\
                   (self._large_high_delist_turnover.reindex(self._large_high_ret.index).fillna(0) + self._small_high_delist_turnover.reindex(self._small_high_ret.index).fillna(0)) * self.context.delist_loss

        low_fee = (self._large_low_turnover.reindex(self._large_low_ret.index).fillna(0) + self._small_low_turnover.reindex(self._small_low_ret.index).fillna(0)) * self.context.fee +\
                   (self._large_low_delist_turnover.reindex(self._large_low_ret.index).fillna(0) + self._small_low_delist_turnover.reindex(self._small_low_ret.index).fillna(0)) * self.context.delist_loss
        
        self._high_ret = (self._large_high_ret + self._small_high_ret) * 0.5
        self._low_ret = (self._large_low_ret + self._small_low_ret) * 0.5

        self.high_ret = self._high_ret - high_fee * 0.5
        self.low_ret = self._low_ret - low_fee * 0.5

        if self.context.factor_direction == 'pos':
            self.diff_ret = self._high_ret - self._low_ret - high_fee * 0.5 - low_fee * 0.5
        else:
            self.diff_ret = self._low_ret - self._high_ret - high_fee * 0.5 - low_fee * 0.5
            
    def _cal_month_ret(self, daily_ret):
        daily_ret = daily_ret.copy()
        daily_ret.index = [x[:-2] for x in daily_ret.index]
        month_ret = daily_ret.groupby(level=0).apply(lambda x:x.add(1).prod() - 1)
        return month_ret
    
    def _get_month_ret(self):
        self.high_month_ret = self._cal_month_ret(self.high_ret)
        self.low_month_ret = self._cal_month_ret(self.low_ret)
        self.diff_month_ret = self._cal_month_ret(self.diff_ret)
        
    def _ols_stats(self, ret, option='nw'):
        if option == 'nw':
            maxlags = int(np.ceil(4 * (len(ret) / 100) ** (2 / 9))) 
            model = sm.OLS(ret.values, np.ones(len(ret))).fit(missing = 'drop', cov_type = 'HAC', cov_kwds={'maxlags':maxlags})
        else:
            model = sm.OLS(ret.values, np.ones(len(ret))).fit()
        return model.tvalues[0], model.pvalues[0]                            
                             
    def _cal_stats(self, high, low, diff, option='nw'):
        t_high, p_high = self._ols_stats(high, option=option)
        t_low, p_low = self._ols_stats(low, option=option)
        t_diff, p_diff = self._ols_stats(diff, option=option)

        return [[t_high, t_low, t_diff],
                 [p_high, p_low, p_diff]]
    
    def _stat_analysis(self):
        month_stat = np.array([[self.high_month_ret.mean(), self.low_month_ret.mean(), self.diff_month_ret.mean()]] +\
                                self._cal_stats(self.high_month_ret, self.low_month_ret, self.diff_month_ret) +\
                                self._cal_stats(self.high_month_ret, self.low_month_ret, self.diff_month_ret, option='non-nw'))
        
        daily_stat = np.array([[self.high_ret.mean(), self.low_ret.mean(), self.diff_ret.mean()]] +\
                                 self._cal_stats(self.high_ret, self.low_ret, self.diff_ret) +\
                                 self._cal_stats(self.high_ret, self.low_ret, self.diff_ret, option='non-nw'))
            
        index = pd.MultiIndex.from_product([[self.context.factor_name], ['mean_ret', 't-value(nw)', 'p-value(nw)', 't-value', 'p-value']])
        columns = pd.MultiIndex.from_product([['monthly', 'daily'], ['high', 'low', 'diff']])
        
        df_stat = pd.DataFrame(np.concatenate((month_stat, daily_stat), axis=1), index=index, columns=columns)
        self.df_stat = df_stat.applymap(lambda x: '%.4f' % x).astype(float)

    def save(self, factor_name=None):
        if factor_name:
            ps.pickle_save(self, self.context.savedir+"{0}.pkl".format(factor_name))
        else:
            ps.pickle_save(self, self.context.savedir+"{0}.pkl".format(self.context.factor_name))


class FactorStat_size(FactorStat):

    def __init__(self, context, cut_item='pb'):
        self.context = context
        self.cut_item = cut_item
        self._get_pos_hold()
        self._get_group_ret()
        self._get_month_ret()
        self._stat_analysis()
        
    def _cal_factor_pos(self, date):
        if self.context.factor_name == 'circ_mv':
            df = self.context.df_factor.loc[date][[self.context.factor_name, self.cut_item]].dropna(axis=0, how='any')
            df['bp'] = 1 / df[self.cut_item]
            del df[self.cut_item]
        df['mv_cut'] = pd.qcut(df['circ_mv'], 2, ['small', 'large'])
        df['factor_cut'] = None
        df.loc[df['mv_cut']  == 'small', 'factor_cut'] = pd.qcut(df.loc[df['mv_cut']  == 'small', 'bp'], [0, 0.3, 0.7, 1], ['low', 'medium', 'high'])
        df.loc[df['mv_cut']  == 'large', 'factor_cut'] = pd.qcut(df.loc[df['mv_cut']  == 'large', 'bp'], [0, 0.3, 0.7, 1], ['low', 'medium', 'high'])
        buy_date = ps.shift_n_trading_day(date, n=-1)
        df = df.loc[list(set(df.index.tolist()) - set(self.context.cannot_buy_dict[buy_date]))]
        df_large_high = df.query('mv_cut=="large" and factor_cut=="high"')
        df_small_high = df.query('mv_cut=="small" and factor_cut=="high"')
        df_large_low = df.query('mv_cut=="large" and factor_cut=="low"')
        df_small_low = df.query('mv_cut=="small" and factor_cut=="low"')
        df_large_medium = df.query('mv_cut=="large" and factor_cut=="medium"')
        df_small_medium = df.query('mv_cut=="small" and factor_cut=="medium"')

        if self.context.weighted_method == 'eql_weighted':
            def cal_weight(df):
                return pd.Series(index = df.index, name='eql_weighted').fillna(1/len(df))
            return cal_weight(df_large_high), cal_weight(df_small_high), cal_weight(df_large_low), cal_weight(df_small_low), cal_weight(df_large_medium), cal_weight(df_small_medium)
    
        if self.context.weighted_method == 'mv_weighted':
            def cal_weight(df):
                return df['circ_mv'] / df['circ_mv'].sum()
            return cal_weight(df_large_high), cal_weight(df_small_high), cal_weight(df_large_low), cal_weight(df_small_low), cal_weight(df_large_medium), cal_weight(df_small_medium)

    def _get_pos_hold(self):
        self.large_high_pos = {}
        self.small_high_pos = {}
        self.large_low_pos = {}
        self.small_low_pos = {}
        self.large_medium_pos = {}
        self.small_medium_pos = {}
        for month_date in sorted(list(self.context.df_ret_dict.keys())):
            buy_date = ps.shift_n_trading_day(month_date, -1)
            self.large_high_pos[buy_date], self.small_high_pos[buy_date], self.large_low_pos[buy_date], self.small_low_pos[buy_date], self.large_medium_pos[buy_date], self.small_medium_pos[buy_date] =\
            self._cal_factor_pos(month_date)
    
    def _get_group_ret(self):
        self._large_high_ret, self._large_high_turnover, self._large_high_delist_turnover = self._cal_ret(self.large_high_pos)
        self._small_high_ret, self._small_high_turnover, self._small_high_delist_turnover = self._cal_ret(self.small_high_pos)
        self._large_low_ret, self._large_low_turnover, self._large_low_delist_turnover = self._cal_ret(self.large_low_pos)
        self._small_low_ret, self._small_low_turnover, self._small_low_delist_turnover = self._cal_ret(self.small_low_pos)
        self._large_medium_ret, self._large_medium_turnover, self._large_medium_delist_turnover = self._cal_ret(self.large_medium_pos)
        self._small_medium_ret, self._small_medium_turnover, self._small_medium_delist_turnover = self._cal_ret(self.small_medium_pos)
        
        small_fee = (self._small_high_turnover.reindex(self._small_high_ret.index).fillna(0) + self._small_medium_turnover.reindex(self._small_medium_ret.index).fillna(0) + self._small_low_turnover.reindex(self._small_low_ret.index).fillna(0)) * self.context.fee +\
                   (self._small_high_delist_turnover.reindex(self._small_high_ret.index).fillna(0) + self._small_medium_delist_turnover.reindex(self._small_medium_ret.index).fillna(0) + self._small_low_delist_turnover.reindex(self._small_low_ret.index).fillna(0)) * self.context.delist_loss

        large_fee = (self._large_high_turnover.reindex(self._large_high_ret.index).fillna(0) + self._large_medium_turnover.reindex(self._large_medium_ret.index).fillna(0) + self._large_low_turnover.reindex(self._large_low_ret.index).fillna(0)) * self.context.fee +\
                   (self._large_high_delist_turnover.reindex(self._large_high_ret.index).fillna(0) + self._large_medium_delist_turnover.reindex(self._large_medium_ret.index).fillna(0) + self._large_low_delist_turnover.reindex(self._large_low_ret.index).fillna(0)) * self.context.delist_loss
        
        self._small_ret = (self._small_high_ret + self._small_medium_ret + self._small_low_ret) / 3
        self._large_ret = (self._large_high_ret + self._large_medium_ret + self._large_low_ret) / 3

        self.small_ret = self._small_ret - small_fee / 3
        self.large_ret = self._large_ret - large_fee / 3

        if self.context.factor_direction == 'pos':
            self.diff_ret = self._small_ret - self._large_ret - small_fee / 3 - large_fee / 3
        else:
            self.diff_ret = self._large_ret - self._small_ret - small_fee / 3 - large_fee / 3
            
    def _get_month_ret(self):
        self.small_month_ret = self._cal_month_ret(self.small_ret)
        self.large_month_ret = self._cal_month_ret(self.large_ret)
        self.diff_month_ret = self._cal_month_ret(self.diff_ret)
                                                     
    def _cal_stats(self, small, large, diff, option='nw'):
        t_small, p_small = self._ols_stats(small, option=option)
        t_large, p_large = self._ols_stats(large, option=option)
        t_diff, p_diff = self._ols_stats(diff, option=option)

        return [[t_small, t_large, t_diff],
                 [p_small, p_large, p_diff]]
    
    def _stat_analysis(self):
        month_stat = np.array([[self.small_month_ret.mean(), self.large_month_ret.mean(), self.diff_month_ret.mean()]] +\
                                self._cal_stats(self.small_month_ret, self.large_month_ret, self.diff_month_ret) +\
                                self._cal_stats(self.small_month_ret, self.large_month_ret, self.diff_month_ret, option='non-nw'))
        
        daily_stat = np.array([[self.small_ret.mean(), self.large_ret.mean(), self.diff_ret.mean()]] +\
                                 self._cal_stats(self.small_ret, self.large_ret, self.diff_ret) +\
                                 self._cal_stats(self.small_ret, self.large_ret, self.diff_ret, option='non-nw'))
            
        index = pd.MultiIndex.from_product([[self.context.factor_name], ['mean_ret', 't-value(nw)', 'p-value(nw)', 't-value', 'p-value']])
        columns = pd.MultiIndex.from_product([['monthly', 'daily'], ['small', 'large', 'diff']])
        
        df_stat = pd.DataFrame(np.concatenate((month_stat, daily_stat), axis=1), index=index, columns=columns)
        self.df_stat = df_stat.applymap(lambda x: '%.4f' % x).astype(float)


class FactorStat_mkt(FactorStat):

    def __init__(self, context):
        self.context = context
        self._get_pos_hold()
        self._get_group_ret()
        self._get_month_ret()
        self._stat_analysis()

    def _get_pos_hold(self):
        self.mkt_pos = {}
        for month_date in sorted(list(self.context.df_ret_dict.keys())):
            buy_date = ps.shift_n_trading_day(month_date, -1)
            self.mkt_pos[buy_date] = self._cal_factor_pos(month_date)
    
    def _cal_factor_pos(self, date):
        df = self.context.df_factor.loc[date][['circ_mv']].dropna(axis=0, how='any')

        if self.context.weighted_method == 'eql_weighted':
            return pd.Series(index = df.index, name='eql_weighted').fillna(1/len(df))

        if self.context.weighted_method == 'mv_weighted':
            return df['circ_mv'] / df['circ_mv'].sum()

    def _get_group_ret(self):
        self._mkt_ret, self._mkt_turnover, self._delist_turnover = self._cal_ret(self.mkt_pos)
        fee = self._mkt_turnover.reindex(self._mkt_ret.index).fillna(0) * self.context.fee + self._delist_turnover.reindex(self._mkt_ret.index).fillna(0) * self.context.delist_loss
        self.mkt_ret = self._mkt_ret - fee
    
    def _get_month_ret(self):
        self.mkt_month_ret = self._cal_month_ret(self.mkt_ret)

    def _cal_stats(self, mkt, option='nw'):
        t_mkt, p_mkt = self._ols_stats(mkt, option=option)
        return [[t_mkt],
                 [p_mkt]]
    
    def _stat_analysis(self):
        month_stat = np.array([[self.mkt_month_ret.mean()]] +\
                                self._cal_stats(self.mkt_month_ret) +\
                                self._cal_stats(self.mkt_month_ret, option='non-nw'))
        
        daily_stat = np.array([[self.mkt_ret.mean()]] +\
                                self._cal_stats(self.mkt_ret) +\
                                self._cal_stats(self.mkt_ret, option='non-nw'))
            
        index = pd.MultiIndex.from_product([['mkt'], ['mean_ret', 't-value(nw)', 'p-value(nw)', 't-value', 'p-value']])
        columns = ['monthly', 'daily']
        
        df_stat = pd.DataFrame(np.concatenate((month_stat, daily_stat), axis=1), index=index, columns=columns)
        self.df_stat = df_stat.applymap(lambda x: '%.4f' % x).astype(float) 

