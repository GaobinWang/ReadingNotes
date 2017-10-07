#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 18:05:43 2017

@author: Lesile
"""

###金融和Python语法
S0 = 100.
K = 105.
T = 1.0
r = 0.05
sigma = 0.2

from numpy import *

I = 100000

random.seed(1000)
z = random.standard_normal(I)
ST = S0 * exp(r * T + sigma * sqrt(T) * z)
hT = maximum(ST - K, 0)
C0 = exp(-r * T) * sum(hT) / I

print("Value of the European Call Option %5.3f" % C0)

###在更短的时间内得到结果
import numpy as np
import pandas as pd
from pandas_datareader import data as web

goog = web.DataReader('GOOG', data_source='google',
                      start='3/14/2009', end='4/14/2010')
goog.index.name = u'Date'
goog.tail()

goog['Log_Ret'] = np.log(goog['Close'] / goog['Close'].shift(1))
goog['Volatility'] = goog['Log_Ret'].rolling(window=252).std() * np.sqrt(252)

%matplotlib inline
goog[['Close', 'Volatility']].plot(subplots=True, color='blue',
                                   figsize=(8, 6), grid=True);
    
###确保高性能
#以实现f函数的2500万次运算为例

#方法一:循环（1 loop, best of 3: 17.7 s per loop）
loops = 25000000
from math import *
a = range(1, loops)
def f(x):
    return 3 * log(x) + cos(x) ** 2
%timeit r = [f(x) for x in a]

#利用Numpy提供的向量化操作（1 loop, best of 3: 1.08 s per loop）
import numpy as np
a = np.arange(1, loops)
%timeit r = 3 * np.log(a) + np.cos(a) ** 2

#numexpr(1 loop, best of 3: 574 ms per loop)
import numexpr as ne
ne.set_num_threads(1)
f = '3 * log(a) + cos(a) ** 2'
%timeit r = ne.evaluate(f)

#使用一个CPU的所有可用线程（1 loop, best of 3: 151 ms per loop）
ne.set_num_threads(4)
%timeit r = ne.evaluate(f)

    