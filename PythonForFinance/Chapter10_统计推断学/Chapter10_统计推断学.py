#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:35:14 2017

@author: Lesile
"""


"""
###Chapter10 统计推断
"""

###生成随机数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##均匀分布随机数
sample_sizes = 10000
rn1 = np.random.rand(sample_sizes)
rn1 = pd.DataFrame(rn1)
rn1.hist(grid = False,bins = 100)

##生成0-10之间的随机数
rn2 = np.random.randint(0,11,sample_sizes)
rn2 = pd.DataFrame(rn2)
rn2.hist(grid = False,bins = 100)

##从一组样本中随机抽样
nums = [3,1,5,10,2]
rn3 = np.random.choice(nums,sample_sizes)
rn3 = pd.DataFrame(rn3)
rn3.hist(grid = False,bins = 100)

##生成随机数
sample_sizes = 10 ** 5
#标准正态分布随机数
rn1 = np.random.standard_normal(sample_sizes)
#均值为100，标准差为20的随机数
rn2 = np.random.normal(100,20,sample_sizes)
#自由度为1的卡方分布
rn3 = np.random.chisquare(df = 1,size = sample_sizes)
#lambda = 1的Possion分布
rn4 = np.random.poisson(lam=1,size = sample_sizes)

##绘制直方图
fig = plt.figure(figsize=(10,8))
plt.subplots(nrows=2, ncols=2)
ax1 = plt.subplot(221)
ax1.hist(rn1,bins = 50)
ax1.set_title("standard normal")
ax1.set_ylabel("Freq")
ax2 = plt.subplot(222)
ax2.hist(rn2,bins = 50)
ax2.set_title("normal")
ax3 = plt.subplot(223)
ax3.hist(rn3,bins = 50)
ax3.set_title("chi")
ax3.set_ylabel("Freq")
ax4 = plt.subplot(224)
ax4.hist(rn3,bins = 50)
ax4.set_title("chi")

"""
###10.2 模拟
"""

"""
###10.4 风险测度
"""

"""
###10.4.1  VaR
"""

###VaR
##Example1
#再次使用BSM设置
S0 = 100 
r = 0.05
sigma = 0.25
T = 30/365.
I = 10 ** 4
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
#计算损失
R_gbm = np.sort(ST - S0)
#绘图
fig = plt.figure(figsize=(10,8))
plt.hist(R_gbm,bins = 100)
plt.xlabel("return")
plt.ylabel("freq")
plt.grid(True)

#计算损失的分位数
import scipy.stats as scs
percs = [0.01,0.1,1.,2.5,5.0,10.0]
var = scs.scoreatpercentile(R_gbm,percs)
print("%16s %16s" % ('Confidence Level','Value-at-Risk'))
print(33*'-')
for pair in zip(percs,var):
    print("%16.2f %16.3f" % (100 - pair[0],-pair[1]))
    
##Example2
#Merton的跳跃扩散(动态模拟)
S0 = 100.
r = 0.05
sigma = 0.25
lamb = 0.75
mu = -0.6
delta = 0.25
T = 1.0
I = 10 ** 5
M= 50

dt = 30./365/M
rj = lamb *(np.exp(mu + 0.5 * delta ** 2) - 1)
S = np.zeros((M + 1,I))
S[0] = S0
sn1 = np.random.standard_normal((M+1,I))
sn2 = np.random.standard_normal((M+1,I))
poi = np.random.poisson(lamb * dt,(M + 1,I))
for t in range(1,M + 1,1):
    S[t] = S[t-1] * (np.exp((r - rj - 0.5 * sigma ** 2)* dt + sigma * np.sqrt(dt) * sn1[t])+(np.exp(mu + delta * sn2[t]) -1) * poi[t])
#计算损失
R_jd = np.sort(S[-1] - S0)
#绘图
fig = plt.figure(figsize=(10,8))
plt.hist(R_jd,bins = 100)
plt.xlabel("return")
plt.ylabel("freq")
plt.grid(True)

#计算损失的分位数
import scipy.stats as scs
percs = [0.01,0.1,1.,2.5,5.0,10.0]
var = scs.scoreatpercentile(R_jd,percs)
print("%16s %16s" % ('Confidence Level','Value-at-Risk'))
print(33*'-')
for pair in zip(percs,var):
    print("%16.2f %16.3f" % (100 - pair[0],-pair[1]))
##两种方法下的VaR的比较
percs = list(np.arange(0.,10.1,0.1))
gbm_var = scs.scoreatpercentile(R_gbm,percs)
jd_var = scs.scoreatpercentile(R_jd,percs)
plt.plot(percs,gbm_var,'b',lw = 1.5,label='GBM')
plt.plot(percs,jd_var,'r',lw = 1.5,label='JD')
plt.xlabel("100-confidence level [%]")
plt.ylabel("value - at -risk")
plt.grid(True)
plt.legend(loc = 4)

"""
###10.4.2   CVaR
"""
#再次使用BSM设置
S0 = 100 
r = 0.05
sigma = 0.25
T = 1.
I = 10 ** 5
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
#考虑固定损失水平L和违约率p
L = 0.5
p = 0.01
#生成违约方案，只能发生一次违约
D = np.random.poisson(p*T,I)
D = np.where(D > 1,1,D)
#如果没有违约，未来指数水平的风险中立值应该等于资产当日现值
np.exp(-r * T) * 1 / I * np.sum(ST)

#我们限定条件下，CVaR的计算
CVaR = np.exp(-r * T) * 1 / I * np.sum(L * D * ST)
CVaR

#CVA的计算
S0_CVA = np.exp(-r * T) * 1 / I * np.sum((1 - L * D) * ST)
S0_CVA

S0_adj = S0 - CVaR
S0_adj

np.count_nonzero(L * D * ST)

plt.hist(L * D * ST, bins=100)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)

#现在看欧式看涨期权的情况
K = 100.
hT = np.maximum(ST - K, 0)
C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
C0

CVaR = np.exp(-r * T) * 1 / I * np.sum(L * D * hT)
CVaR

C0_CVA = np.exp(-r * T) * 1 / I * np.sum((1 - L * D) * hT)
C0_CVA

np.count_nonzero(L * D * hT)  # number of losses


np.count_nonzero(D)  # number of defaults

I - np.count_nonzero(hT)  # zero payoff

plt.hist(L * D * hT, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)
plt.ylim(ymax=350)



