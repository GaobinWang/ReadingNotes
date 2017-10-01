#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:53:27 2017

@author: Lesile
"""

"""
本章主要介绍matplotlib库的基本可视化功能
#2D绘图
#3D绘图
"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

###2维绘图
"""
pyplot子库中的plot函数是最基础的绘图函数，但是也相当强大。原则上他需要2组数值：
#x值
#y值
当然x和y值的数量必须相等
"""
np.random.seed(1000)
y = np.random.standard_normal(20)
x = range(len(y))
plt.plot(x,y)

plt.plot(y) #以索引值为对应的x
plt.plot(y.cumsum())

"""
matplotlib提供了大量函数以自定义绘图样式
例如操纵坐标轴、增加网格线、添加标签等
"""
plt.plot(y.cumsum())
plt.grid(True) #添加网格线
plt.axis('tight') #紧凑坐标轴
#可以使用plt.xlim和plt.ylim设置每个坐标轴的最小值和最大值
plt.plot(y.cumsum())
plt.grid(True) #添加网格线
plt.xlim(-1,20)
plt.ylim(np.min(y.cumsum()) - 1,
         np.max(y.cumsum()) + 1)

