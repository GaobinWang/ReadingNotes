# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:15:24 2019

@author: Lesile
"""
import pandas as pd
import numpy as np


"""
datetime模块中的数据类型
date 以公历形式存储日期(年月日)
time 将时间存储为时、分、秒、毫秒
datetime 存储时间和日期
timedelta 表示两个datetime值之间的差(日 秒 毫秒)
"""

from datetime import datetime
now = datetime.now()
now.year
now.month
now.day
now.hour
now.minute
now.second
now.microsecond

delta = datetime(2019,5,1) - datetime(2019,4,20,8,15,1,20)
delta.days
delta.seconds
delta.microseconds

##字符串和datetime的相互转换
stamp = datetime(2019,5,1)
str(stamp)
stamp.strftime("%Y-%m-%d")

value = "2019-5-01"
datetime.strptime(value,"%Y-%m-%d")


###可以使用dateutil这个第三方包中的parser.parse方法将字符串转化为日期
from dateutil.parser import parse
parse(value)

#日期出现在月的前面时dayfirst = True
parse("6/12/2011",dayfirst = True)
parse("6/12/2011",dayfirst = False)

###pandas可以处理多种不同的日期格式 pd.to_datetime

pd.to_datetime(["7/6/2011","8/6/2011",None])
#NaT(Not a Time)是pandas中时间戳数据的NA值



"""
datetime格式定义
%w 用整数表示星期几[0,6]
%U 每年的第几周[0,53],星期天被认为是每周的第一天
%W 每年的第几周[0,53],星期一被认为是每周的第一天
%F %Y-%m-%d 简写形式
"""

now.strftime("%F")


###时间序列基础





#跟其他Series一样,不同索引的时间序列之间的算术运算会自动按日期对齐
#pandas用NumPy的datetime64数据类型以纳秒形式存储时间戳
#只要有需要，TimeStamp可以随时自动转换成datetime对象


datelist = [datetime(2011,1,1),datetime(2011,1,2),datetime(2011,1,3)]


ts = pd.Series(np.random.randn(3),index =datelist)

stamp = ts.index[0]



###索引 选取 子集的构造
longer_ts = pd.Series(np.random.normal(size = (1,1000))[0],index = pd.date_range("1/1/2000",periods = 1000))
longer_ts["2001"]
longer_ts["2001-05"]
longer_ts[datetime(2001,1,7):]
#上面这些操作对DataFrame也有效
longer_ts = pd.DataFrame(np.random.normal(size = (1000,3)),index = pd.date_range("1/1/2000",periods = 1000))

longer_ts["2001"]
longer_ts["2001-05"]
longer_ts[datetime(2001,1,7):]


###带有重复索引的时间序列
dates = pd.DatetimeIndex(["1/1/2000","1/2/2000","1/2/2000","1/2/2000","1/3/2000"])
dup_ts = pd.Series(np.arange(5),index =dates)

dup_ts.index.is_unique #检查索引是否唯一

#假设你要对具有非唯一时间戳数据进行聚合,一个办法是使用groupby，并传入level = 0(索引的唯一一层)

groupd = dup_ts.groupby(level = 0).mean()


###生成日期范围
dates = pd.date_range("2000-01-01","2000-12-31",freq = "BM") #BM business end of month 每月最后一个工作日

###频率和日期偏移量
"""
pandas中的频率是由一个基础频率和一个乘数组成的，
基础频率通常以一个字符串别名表示,比如“M”表示每月,"H"表示每小时
基础频率都有一个被称为日期偏移量(date offset)的对象与之对应
"""

from pandas.tseries.offsets import Hour,Minute
hour = Hour()
four_hours = Hour(4)

dates = pd.date_range(start = "2000-01-01",end = "2000-12-31",freq = "4h")



dates = pd.date_range(start = "2000-01-01",end = "2000-12-31",freq = Hour(1) + Minute(30))
dates = pd.date_range(start = "2000-01-01",end = "2000-12-31",freq = "1h30min")


"""
时间序列的基础频率
D 每日历日
B 每工作日
M 每月最后一个日历日
BM 每月最后一个工作日
MS 每月第一个日历日
BMS 每月第一个工作日
W 每周星期日
W-MON 从指定的星期几开始算起,每周
WOM-3Fri 每月第三个星期五
"""



dates = pd.date_range(start = "2000-01-01",end = "2000-12-31",freq = "M")
dates = pd.date_range(start = "2000-01-01",end = "2000-12-31",freq = "WOM-3Fri")

###移动(超前和滞后)数据
#Series和DataFrame都有一个shift方法用于执行单纯的前移或者后移操作,保持索引不变
ts = pd.Series(np.random.randn(4),index = pd.date_range("2000-01-01",periods = 4,freq = "M"))
ts
ts.shift(2)
ts.shift(-2)

ts/ts.shift(1)-1
ts.pct_change()

#单纯的位移操作不会修改索引,所以部分数会被丢弃.因此，如果频率已知,则可以将其传给shift以便实现对时间戳进行位移
ts.shift(2,freq = "M")
ts.shift(3,freq = "D")
ts.shift(1,freq = "3D")









