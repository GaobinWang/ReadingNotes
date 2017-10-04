#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:57:48 2017

@author: Lesile
"""

"""
Chapter8 高性能的Python
“不要降低预期去屈就性能，而要提升性能满足预期” 
对于性能关键应用，应该总是检查两件事情：是否使用正确的实现范型；是否使用正确的高性能库。
许多高性能库可以用于加速Python代码的的执行，下面几个库最为常用:
    Cython：用于合并Python和C语言静态编译范型；
    IPython.parallel：用于在本地或者集群上并行执行代码/函数；
    numexpr：用于快速数值计算；
    multiprocessing：Python内建的(本地)并行处理模块；
    Numba：用于为CPU动态编译Python代码；
    NumbaPro：用于为多核CPU和GPU动态编译Python代码。
"""

###用于比较不同代码性能的函数
def perf_comp_data(func_list, data_list, rep=3, number=1):
    ''' Function to compare the performance of different functions.
    
    Parameters
    ==========
    func_list : list
        list with function names as strings
    data_list : list
        list with data set names as strings
    rep : int
        number of repetitions of the whole comparison
    number : int
        number of executions for every function
    '''
    from timeit import repeat
    res_list = {}
    for name in enumerate(func_list):
        stmt = name[1] + '(' + data_list[name[0]] + ')'
        setup = "from __main__ import " + name[1] + ', ' \
                                    + data_list[name[0]]
        results = repeat(stmt=stmt, setup=setup,
                         repeat=rep, number=number)
        res_list[name[1]] = sum(results) / rep
    res_sort = sorted(res_list.items(),
                      key=lambda x: (x[1], x[0]))
    for item in res_sort:
        rel = item[1] / res_sort[0][1]
        print ('function: ' + item[0] +
              ', av. time sec: %9.5f, ' % item[1]
            + 'relative: %6.1f' % rel)
        
"""
###8.1 python范型与性能
"""
from math import *
def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)

I = 500000
a_py = range(I)

def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res

import numba as nb
f11 = nb.jit(f1)

def f2(a):
    return [f(x) for x in a]

def f3(a):
    ex = 'abs(cos(x)) ** 0.5 + sin(2 + 3 * x)'
    return [eval(ex) for x in a]

import numpy as np
a_np = np.arange(I)
def f4(a):
    return (np.abs(np.cos(a)) ** 0.5 + np.sin(2 + 3 * a))

import numexpr as ne

def f5(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)

def f6(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(16)
    return ne.evaluate(ex)

%%time
r1 = f1(a_py)
r11 = f11(a_py)
r2 = f2(a_py)
r3 = f3(a_py)
r4 = f4(a_np)
r5 = f5(a_np)
r6 = f6(a_np)

np.allclose(r1, r2)
np.allclose(r1, r11)
np.allclose(r1, r3)
np.allclose(r1, r4)
np.allclose(r1, r5)
np.allclose(r1, r6)

func_list = ['f1', 'f11','f2', 'f3', 'f4', 'f5', 'f6']
data_list = ['a_py', 'a_py','a_py', 'a_py', 'a_np', 'a_np', 'a_np']
perf_comp_data(func_list, data_list)
#为何f11没有显著提升(numba)

"""
###8.2 内存布局与性能
"""
import numpy as np
np.zeros((3, 3), dtype=np.float64, order='C')

c = np.array([[ 1.,  1.,  1.],
              [ 2.,  2.,  2.],
              [ 3.,  3.,  3.]], order='C')

f = np.array([[ 1.,  1.,  1.],
              [ 2.,  2.,  2.],
              [ 3.,  3.,  3.]], order='F')

x = np.random.standard_normal((3, 150000))
C = np.array(x, order='C')
F = np.array(x, order='F')
x = 0.0

get_ipython().magic('timeit C.sum(axis=0)')
get_ipython().magic('timeit C.sum(axis=1)')

get_ipython().magic('timeit C.std(axis=0)')
get_ipython().magic('timeit C.std(axis=1)')

get_ipython().magic('timeit F.sum(axis=0)')
get_ipython().magic('timeit F.sum(axis=1)')

get_ipython().magic('timeit F.std(axis=0)')
get_ipython().magic('timeit F.std(axis=1)')


"""
###8.3 并行计算
"""

from random import gauss
#期权的蒙特卡洛估值方法
def bsm_mcs_valuation(strike):
    ''' Dynamic Black-Scholes-Merton Monte Carlo estimator for European calls.
    
    Parameters
    ==========
    strike: float
        strike price of the option
    
    Returns
    =======
    value : float
        estimate for present value of the European call option
    '''
    # Parameter Values
    S0 = 100.  # initial index level
    T = 1.0  # time-to-maturity
    r = 0.05  # riskless short rate
    vola = 0.2  # volatility
    M = 50  # number of time steps
    I = 20000  # number of simulations
    dt = T/M
    # Valuation Algorithm
    rand = np.random.standard_normal((M + 1,I))  # pseudorandom numbers
    S = np.zeros((M + 1,I))
    S[0] = S0
    for t in range(1,M + 1):
        S[t] = S[t-1] * np.exp((r - 0.5 * vola ** 2) * dt + vola * np.sqrt(dt) * rand[t])
    # index values at maturity
    hT = np.maximum(S[-1] - strike, 0)  # inner values at maturity
    value = np.exp(-r * T) * np.sum(hT) / I  # Monte Carlo estimator
    return(value)
    
bsm_mcs_valuation(100.)

##顺序化计算
def seq_value(n):
    strikes = np.linspace(80,120,n)
    option_values = []
    for strike in strikes:
        option_values.append(bsm_mcs_valuation(strike))
    return strikes,option_values

n = 100
%time strikes,option_strikes_seq = seq_value(n)


import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize = (10,5))
plt.plot(strikes,option_strikes_seq)
plt.xlabel("strikes")
plt.ylabel("European call option values")

###并行计算
from  ipyparallel import Client
c = Client(profile = "default")
view = c.load_balanced_view()

def par_value(n):
    strikes = np.linspace(80,120,n)
    option_values = []
    for strike in strikes:
        value = view.apply_async(bsm_mcs_valuation,strike)
        option_values.append(value)
    c.wait(option_values)
    return strikes,option_values
%time strikes,option_strikes_seq = par_value(n)


"""
###8.4 多处理
"""
import multiprocessing as mp
import math
def simulate_geometric_brownian_motion(p):
    M, I = p
      # time steps, paths
    S0 = 100; r = 0.05; sigma = 0.2; T = 1.0
      # model parameters
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                    sigma * math.sqrt(dt) * np.random.standard_normal(I))
    return paths

paths = simulate_geometric_brownian_motion((5, 2))
paths
I = 10000  # number of paths
M = 50  # number of time steps
t = 20  # number of tasks/simulations
# running on server with 4 cores
from time import time
times = []
for w in range(1, 5):
    t0 = time()
    pool = mp.Pool(processes=w)
      # the pool of workers
    result = pool.map(simulate_geometric_brownian_motion, t * [(M, I), ])
      # the mapping of the function to the list of parameter tuples
    times.append(time() - t0)
    
import matplotlib.pyplot as plt
plt.plot(range(1, 5), times)
plt.plot(range(1, 5), times, 'ro')
plt.grid(True)
plt.xlabel('number of processes')
plt.ylabel('time in seconds')
plt.title('%d Monte Carlo simulations' % t)
# tag: multi_proc
# title: Comparison execution speed dependent on the number of threads used (4 core machine)
# size: 60

"""
###8.5 动态编译
Numba是开源、Numpy感知的优化Python代码的编译器。它使用LLVM编译器基础构架，将Python字节代码
编译专门用于Numpy运行时和Scipy模块的机器代码。
"""

##############################
#一个简单的示例
##############################
#示例1：包含嵌套循环的算法
from math import cos,log
def f_py(I,J):
    res = 0
    for i in range(I):
        for j in range(J):
            res += int(cos(log(1)))
    return(res)
I,J = 5000,5000
%timeit f_py(I,J)

#示例2：向量化
def f_np(I,J):
    a = np.ones((I,J),dtype = np.float64)
    return int(np.sum(np.cos(np.log(a)))),a
%time res,a = f_np(I,J)
#这种方法快得多(大约8-10倍)，但是并不能真正高效的利用内存。ndarray对象消耗200MB内存
#考虑到RAM的数量，很容易选择I和J使Numpy方法变得不可行
a.nbytes

#示例3：Numba
import numba as nb
f_nb = nb.jit(f_py)
%time f_nb(I,J)

##性能比较
func_list = ['f_py', 'f_np', 'f_nb']
data_list = 3 * ['I, J']
perf_comp_data(func_list, data_list)

##############################
#二项式期权定价方法
##############################
from numpy import exp,sqrt

# model & option Parameters
S0 = 100.  # initial index level
T = 1.  # call option maturity
r = 0.05  # constant short rate
vola = 0.20  # constant volatility factor of diffusion

# time parameters
M = 1000  # time steps
dt = T / M  # length of time interval
df = exp(-r * dt)  # discount factor per time interval

# binomial parameters
u = exp(vola * sqrt(dt))  # up-movement
d = 1 / u  # down-movement
q = (exp(r * dt) - d) / (u - d)  # martingale probability

import numpy as np
def binomial_py(strike):
    ''' Binomial option pricing via looping.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    # LOOP 1 - Index Levels
    S = np.zeros((M + 1, M + 1), dtype=np.float64)
      # index level array
    S[0, 0] = S0
    z1 = 0
    for j in range(1, M + 1, 1):
        z1 = z1 + 1
        for i in range(z1 + 1):
            S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))
            
    # LOOP 2 - Inner Values
    iv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # inner value array
    z2 = 0
    for j in range(0, M + 1, 1):
        for i in range(z2 + 1):
            iv[i, j] = max(S[i, j] - strike, 0)
        z2 = z2 + 1
        
    # LOOP 3 - Valuation
    pv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # present value array
    pv[:, M] = iv[:, M]  # initialize last time point
    z3 = M + 1
    for j in range(M - 1, -1, -1):
        z3 = z3 - 1
        for i in range(z3):
            pv[i, j] = (q * pv[i, j + 1] +
                        (1 - q) * pv[i + 1, j + 1]) * df
    return pv[0, 0]

%time round(binomial_py(100), 3)


def binomial_np(strike):
    ''' Binomial option pricing with NumPy.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    # Index Levels with NumPy
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md
    
    # Valuation Loop
    pv = np.maximum(S - strike, 0)

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        pv[0:M - z, t] = (q * pv[0:M - z, t + 1]
                        + (1 - q) * pv[1:M - z + 1, t + 1]) * df
        z += 1
    return pv[0, 0]

M = 4  # four time steps only
mu = np.arange(M + 1)
mu
mu = np.resize(mu, (M + 1, M + 1))
mu
md = np.transpose(mu)
md
mu = u ** (mu - md)
mu.round(3)
md = d ** md
md.round(3)
S = S0 * mu * md
S.round(3)
M = 1000  # reset number of time steps
%time round(binomial_np(100), 3)

binomial_nb = nb.jit(binomial_py)
%time round(binomial_nb(100), 3)

func_list = ['binomial_py', 'binomial_np', 'binomial_nb']
K = 100.
data_list = 3 * ['K']
perf_comp_data(func_list, data_list)


"""
###8.6 用Cython进行静态编译
"""
#方法一
def f_py(I, J):
    res = 0.  # we work on a float object
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res

I, J = 500, 500
%time f_py(I, J)

#方法二
import numba as nb
f_nb = nb.jit(f_py)
%time f_nb(I, J)

#方法三
import pyximport
pyximport.install()
import sys
from nested_loop import f_cy
#上面的方法报错了


%load_ext Cython
%%cython
#
# Nested loop example with Cython
#
def f_cy(int I, int J):
    cdef double res = 0
    # double float much slower than int or long
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res

%time res = f_cy(I, J)

func_list = ['f_py', 'f_cy', 'f_nb']
I, J = 500, 500
data_list = 3 * ['I, J']

func_list = ['f_py', 'f_nb']
I, J = 500, 500
data_list = 2 * ['I, J']
perf_comp_data(func_list, data_list)

"""
###8.7 在GPU上生成随机数
有一个金融领域可以从GPU的使用中得到很大好处：蒙特卡洛模拟，特别是随机数的生成。
"""
import numpy as np
from numbapro.cudalib import curand


###numba
import os
import numba
#@numba.jit()
def c(n):
	count=0
	for i in range(n):
		for i in range(n):
			count+=1
	return count

n=99999
c(n)

