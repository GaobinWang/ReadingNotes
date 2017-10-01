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

###一维数据集
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
#
plt.figure(figsize = (7,4)) #设置图像的尺寸
plt.plot(y.cumsum(), 'b:', lw = 1.5)
plt.plot(y.cumsum(), 'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
"""
#标准颜色缩写
b:蓝
g:绿
r:红
c:青
m:品红
y:黄
k:黑
w:白色
#标准样式字符
-:实线样式
--:短划线样式
-.:点实线样式
::虚线样式
o:圆标记
v:向下三角形标记
+:加好标记
"""

###二维数据集
"""
二维数据集画图涉及到两个问题:
    第一，两列希望画出不同的图形；
    第二，两列的刻度不一样
"""
np.random.seed(2000)
y = np.random.standard_normal((20,2)).cumsum(axis = 0)

#图一
plt.figure(figsize = (7,4)) #设置图像的尺寸
plt.plot(y,lw = 1.5)
plt.plot(y, 'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

#图二:加图例
plt.figure(figsize = (7,4)) #设置图像的尺寸
plt.plot(y[:,0],lw = 1.5,label = '1st')
plt.plot(y[:,1],lw = 1.5,label = '2st')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
"""
#plt.legend选项
空白:自动
0:最佳
1:右上
2:左上
3:左下
4:右下
5:右
...
"""

#图三:数值不协调
y[:,0] = y[:,0] * 100
plt.figure(figsize = (7,4)) #设置图像的尺寸
plt.plot(y[:,0],lw = 1.5,label = '1st')
plt.plot(y[:,1],lw = 1.5,label = '2st')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

"""
解决上述问题的方法有两个:
    第一，使用2个y轴(左、右)
    第二，使用两个子图(上、下；左、右)
"""
#图四:双坐标轴
fig,ax1 = plt.subplots()
plt.plot(y[:,0],'b',lw = 1.5,label = '1st(Left)')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc = 8)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
ax2 = ax1.twinx()
plt.plot(y[:,1],'g',lw = 1.5,label = '2st(Right)')
plt.plot(y[:,1], 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
"""
管理坐标轴的代码行是关键
fig,ax1 = plt.subplots() #使用左轴画第一个数据
ax2 = ax1.twinx() #使用右轴画第二个数据
"""

#图四:两个单独子图
plt.figure(figsize = (7,4)) #设置图像的尺寸
plt.subplot(211)
plt.plot(y[:,0],'b',lw = 1.5,label = '1st')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.subplot(212)
plt.plot(y[:,1],'g',lw = 1.5,label = '2st')
plt.plot(y[:,1], 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
"""
plt.figure有三个整数参数(可以有逗号分隔，也可能没有):
    numrows:指定行数
    numcols:指定列数
    fignum:指定子图编号，从1到numrows*numcols

"""
#图五:两个单独子图+不同绘图格式
plt.figure(figsize = (9,4)) #设置图像的尺寸
plt.subplot(121)
plt.plot(y[:,0],'b',lw = 1.5,label = '1st')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.subplot(122)
plt.bar(np.arange(len(y)),y[:,1],width = 0.5,color='g',label='2nd')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('2nd Data Set')

#图六:两个坐标轴+不同绘图格式
fig,ax1 = plt.subplots()
plt.plot(y[:,0],'b',lw = 1.5,label = '1st')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
ax2 = ax1.twinx()
plt.bar(np.arange(len(y)),y[:,1],width = 0.5,color='g',label='2nd')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('2nd Data Set')

###其他绘图样式:散点图
y = np.random.standard_normal((1000,2))
plt.figure(figsize = (7,5))
plt.plot(y[:,0],y[:,1],'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2st')
plt.title('Scatter Plot')

#scatter函数
plt.figure(figsize = (7,5))
plt.scatter(y[:,0],y[:,1],marker = 'o')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2st')
plt.title('Scatter Plot')

#scatter函数:第三维数据可视化
c = np.random.randint(0,10,len(y))
plt.figure(figsize = (7,5))
plt.scatter(y[:,0],y[:,1],c = c,marker = 'o')
plt.colorbar()
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2st')
plt.title('Scatter Plot')

###其他绘图样式:直方图
plt.figure(figsize = (7,4))
plt.hist(y,label = ['1st','2nd'],bins = 25)
plt.grid(True)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Histogram')

###其他绘图样式:箱形图
fig,ax = plt.subplots(figsize = (7,4))
plt.boxplot(y)
plt.grid(True)
plt.setp(ax,xticklabels = ['1st','2nd'])
plt.xlabel('data set')
plt.ylabel('value')
plt.title('Boxplot')

###金融学图表
start = (2014,5,1)
end = (2014,6,30)


###3D绘图
#将两个一维数组转换为二维数组
strike = np.linspace(50,150,24)
ttm = np.linspace(0.5,2.5,24)
strike,ttm = np.meshgrid(strike,ttm)
#生成第三维数组
iv = (strike - 100) ** 2/(100 * strike)/ttm
#画图
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (9,6))
ax = Axes3D(fig)

surf = ax.plot_surface(strike, ttm, iv, rstride = 2, cstride = 2,
                       cmap = plt.cm.coolwarm,
                       linewidth = 0.5,
                       antialiased = True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf, shrink = 0.5, aspect = 5)