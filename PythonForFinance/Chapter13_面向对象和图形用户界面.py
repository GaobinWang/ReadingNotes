#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:28:47 2017

@author: Lesile
"""

"""
###Chapter13 面向对象和图形用户界面

首先解决问题，然后编写代码 -- Jon Johnson

面向对象技术既有爱好者,又有批评者。

对于真正理解手上问题的程序员，或者从抽象和概括中所得甚多的时候最有益处。
如果不能准确的知道要做什么，更交互式和探索性的编程风格(如过程式编程)可能是更好的选择。

在构建图形用户界面(GUI)时,面向对象方法通常是必不可少的。
"""

"""
###13.1 面向对象
面向对象编程(OOP)是一种编程范型,将概念表现为具有数据字段(描述对象属性)和相关过程(称为方法)的"对象",
对象通常是类的实例,通过类的相互作用设计应用程序和计算机软件.
"""

"""
###13.1.1 Python类基础知识
"""
#定义一个类
class ExampleOne(object):
    pass
#类的实例
c = ExampleOne()
type(c)
dir(c)

#定义一个具有2个属性的类。
#为此，我们定义一个特殊的方法init,每次实例化类的时候自动化调用
#注意,在下面定义的类中,2个属性在内部(即类定义中)分别通过self.a和self.b引用
class ExampleTwo(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b

c = ExampleTwo(1,'text')
#访问对象c的属性值
c.a
c.b
#为属性赋新值
c.a = 100
c.a

#Python在类和对象的使用上颇为灵活:对象的属性甚至可以在实例化之后定义
c = ExampleOne()
c.first_name = 'Jason'
c.last_name = 'Bourne'
c.movies = 4

print(c.first_name,c.last_name,c.movies)

#下面的类定义引入类方法
class ExampleThree(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def addition(self):
        return self.a + self.b
c = ExampleThree(10,15)
c.addition()
c.a += 10
c.addition()

#面向对象编程的优势之一是可重用性
#类的继承(子类 父类)
class ExampleFour(ExampleTwo):
    def addition(self):
        return self.a + self.b
    
c = ExampleFour(10,100)
c
c.addition()

#Python允许多重继承，然而应当小心处理可读性和可维护性
class ExampleFive(ExampleFour):
    def mul(self):
        return self.a * self.b
    
c = ExampleFive(111,333)
c
c.mul()
c.addition()

#类/对象可能需要私有属性
#私有属性通常由一个或多个前导的下划线指明
class ExampleSeven(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.__sum = a + b
    def add(self):
        return self.a + self.b
    def mul(self):
        return self.a * self.b
c = ExampleSeven(6,7)
c.add()
c.mul()
c.a
c.b
c.__sum #私有属性不能直接访问
c._ExampleSeven__sum

##iter方法
name_list = ['Sandra','Lilli','Guido','Zorro','Henry']
for name in name_list:
    print(name)

class sorted_list(object):
    def __init__(self,elements):
        self.elements = sorted(elements)
    def __iter__(self):
        self.position = -1
        return self
    def next(slef):
        if self.position == len(self.elements) - 1:
            raise StopIteration
        self.position += 1
        return self.elements[self.position]

sorted_name_list = sorted_list(name_list)
for name in sorted_name_list:
    print(name)


"""
###13.1.2 简单的短期利率类(Simple Short Rate Class)
"""
#计算折现因子
import numpy as np
def discount_factor(r, t):
    ''' Function to calculate a discount factor.
    
    Parameters
    ==========
    r : float
        positive, constant short rate
    t : float, array of floats
        future date(s), in fraction of years;
        e.g. 0.5 means half a year from now
    
    Returns
    =======
    df : float
        discount factor
    '''
    df = np.exp(-r * t)
      # use of NumPy universal function for vectorization
    return df


import matplotlib.pyplot as plt
%matplotlib inline

t = np.linspace(0, 5)
for r in [0.01, 0.05, 0.1]:
    plt.plot(t, discount_factor(r, t), label='r=%4.2f' % r, lw=1.5)
plt.xlabel('years')
plt.ylabel('discount factor')
plt.grid(True)
plt.legend(loc=0)

#给定利率水平的情况下,计算折现因子
class short_rate(object):
    ''' Class to model a constant short rate object.
    
    Parameters
    ==========
    name : string
        name of the object
    rate : float
        positive, constant short rate
    
    Methods
    =======
    get_discount_factors :
        returns discount factors for given list/array
        of dates/times (as year fractions)
    '''
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate
    def get_discount_factors(self, time_list):
        ''' time_list : list/array-like '''
        time_list = np.array(time_list)
        return np.exp(-self.rate * time_list)

sr = short_rate('r', 0.05)
sr.name, sr.rate
time_list = [0.0, 0.5, 1.0, 1.25, 1.75, 2.0]  # in year fractions
sr.get_discount_factors(time_list)

for r in [0.025, 0.05, 0.1, 0.15]:
    sr.rate = r
    plt.plot(t, sr.get_discount_factors(t),
             label='r=%4.2f' % sr.rate, lw=1.5)
plt.xlabel('years')
plt.ylabel('discount factor')
plt.grid(True)
plt.legend(loc=0)

sr.rate = 0.05
cash_flows = np.array([-100, 50, 75])
time_list = [0.0, 1.0, 2.0]

disc_facts = sr.get_discount_factors(time_list)
disc_facts
# present values
disc_facts * cash_flows
np.sum(disc_facts * cash_flows)
sr.rate = 0.15
np.sum(sr.get_discount_factors(time_list) * cash_flows)

"""
###13.1.3 现金流序列类 Cash Flow Series Class
"""

class cash_flow_series(object):
    ''' Class to model a cash flows series.
    
    Attributes
    ==========
    name : string
        name of the object
    time_list : list/array-like
        list of (positive) year fractions
    cash_flows : list/array-like
        corresponding list of cash flow values
    short_rate : instance of short_rate class
        short rate object used for discounting
    
    Methods
    =======
    present_value_list :
        returns an array with present values
    net_present_value :
        returns NPV for cash flow series
    '''
    def __init__(self, name, time_list, cash_flows, short_rate):
        self.name = name
        self.time_list = time_list
        self.cash_flows = cash_flows
        self.short_rate = short_rate
    def present_value_list(self):
        df = self.short_rate.get_discount_factors(self.time_list)
        return np.array(self.cash_flows) * df
    def net_present_value(self):
        return np.sum(self.present_value_list())

sr.rate = 0.05
cfs = cash_flow_series('cfs', time_list, cash_flows, sr)
cfs.cash_flows
cfs.time_list
cfs.present_value_list()
cfs.net_present_value()
class cfs_sensitivity(cash_flow_series):
    def npv_sensitivity(self, short_rates):
        npvs = []
        for rate in short_rates:
            sr.rate = rate
            npvs.append(self.net_present_value())
        return np.array(npvs)
		
cfs_sens = cfs_sensitivity('cfs', time_list, cash_flows, sr)
short_rates = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
npvs = cfs_sens.npv_sensitivity(short_rates)
npvs

plt.plot(short_rates, npvs, 'b')
plt.plot(short_rates, npvs, 'ro')
plt.plot((0, max(short_rates)), (0, 0), 'r', lw=2)
plt.grid(True)
plt.xlabel('short rate')
plt.ylabel('net present value')

"""
###13.2 图形用户界面
traits库通常用于在现有类的基础上快速构建GUI，很少用于复杂的应用程序
"""
import numpy as np
import traits.api as trapi

class short_rate(trapi.HasTraits):
    name = trapi.Str
    rate = trapi.Float
    time_list = trapi.Array(dtype = np.float,shape = (5,))
    def get_discount_factors(self):
        return np.exp(- self.rate * self.time_list)
    
sr = short_rate()
sr.configure_traits()
#报错:RuntimeError: Invalid Qt API 'pyqt5', valid values are: 'pyqt' or 'pyside'











