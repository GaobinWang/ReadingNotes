#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 19:21:07 2017

@author: Lesile
"""

"""
###Chapter3 入门示例
"""
import os


"""
###3.1 隐含波动率
"""
os.chdir("E:\\Github\\ReadingNotes\\PythonForFinance\\Chapter3_入门示例")

V0 = 17.6639
r = 0.01
import pandas as pd
h5 = pd.HDFStore('vstoxx_data_31032014.h5', 'r')
futures_data = h5['futures_data']  # VSTOXX futures data
options_data = h5['options_data']  # VSTOXX call option data
h5.close()

import datetime as dt
futures_data['DATE'] = futures_data['DATE'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
futures_data['MATURITY'] = futures_data['MATURITY'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
futures_data

options_data['DATE'] = options_data['DATE'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
options_data['MATURITY'] = options_data['MATURITY'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
options_data.info()

options_data['IMP_VOL'] = 0.0
  # new column for implied volatilities
  
from bsm_functions import *
tol = 0.5  # tolerance level for moneyness
for option in options_data.index:
    # iterating over all option quotes
    forward = futures_data[futures_data['MATURITY'] == \
                options_data.loc[option]['MATURITY']]['PRICE'].values[0]
      # picking the right futures value
    if (forward * (1 - tol) < options_data.loc[option]['STRIKE']
                             < forward * (1 + tol)):
        # only for options with moneyness within tolerance
        imp_vol = bsm_call_imp_vol(
                V0,  # VSTOXX value 
                options_data.loc[option]['STRIKE'],
                options_data.loc[option]['TTM'],
                r,   # short rate
                options_data.loc[option]['PRICE'],
                sigma_est=2.,  # estimate for implied volatility
                it=100)
        options_data.ix[option, 'IMP_VOL'] = imp_vol
    

"""
###3.2 蒙特卡洛模拟
"""
from bsm_functions import bsm_call_value
S0 = 100.
K = 105.
T = 1.0
r = 0.05
sigma = 0.2
bsm_call_value(S0, K, T, r, sigma)

###pure python
%run mcs_pure_python.py
#Duration in Seconds    27.246


sum_val = 0.0
for path in S:
    # C-like iteration for comparison
    sum_val += max(path[-1] - K, 0)
C0 = exp(-r * T) * sum_val / I
round(C0, 3)

###Vectorization with NumPy
%run mcs_vector_numpy.py
#Duration in Seconds     0.778

###Full Vectorization with Log Euler Scheme
%run mcs_full_vector_numpy.py
#Duration in Seconds     0.844

###图形化分析
import matplotlib.pyplot as plt
plt.plot(S[:, :10])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')
# tag: index_paths
# title: The first 10 simulated index level paths

"""
###3.3 技术分析
"""
import os
import numpy as np
import pandas as pd
import scipy as sp 
import pandas_datareader.data as web
#该子库包含DataReader函数,这个函数能够帮助我们从不同来源，特别是流行的雅虎财经网站上获取金融事件序列数据。
#由于pandas_datareader库无法从雅虎财经下载数据，我们换用上证综指数据进行演示

#读取数据
os.chdir("E:\\Github\\ReadingNotes\\PythonForFinance\\Chapter3_入门示例")
sz = pd.read_excel("上证综指.xlsx",index_col = 0)

#绘图
sz['close'].plot(grid = True,figsize = (10,6))

#计算移动平均 并作图
sz['mv42'] = pd.rolling_mean(sz['close'],42)
sz['mv252'] = pd.rolling_mean(sz['close'],252)
sz[['close','mv42','mv252']].plot(grid = True,figsize = (10,6))

"""
投资策略:
    买入信号(多头):42天趋势第一次高于252天趋势SD点
    等待(待币):42天趋势在252天趋势的+/-SD个点范围内
    卖出信号(空头):42天趋势线第一低于252天趋势线SD点
"""
sz['mv42-mv252'] = sz['mv42'] - sz['mv252']
sz['mv42-mv252'].tail()
sz['mv42-mv252'].plot()

#编写投资策略
SD = 50
sz['Regime'] = np.where(sz['mv42-mv252'] > SD,1,0)
sz['Regime'] = np.where(sz['mv42-mv252'] < -SD,-1,sz['Regime'])
sz['Regime'].value_counts()
sz['Regime'].tail()
sz['Regime'].plot(lw = 1.5)

#回测
sz['Market'] = np.log(sz['close']/sz['close'].shift(1))
sz['Strategy'] = sz['Regime'].shift(1)*sz['Market']
sz[['Market','Strategy']].cumsum().apply(np.exp).plot()






