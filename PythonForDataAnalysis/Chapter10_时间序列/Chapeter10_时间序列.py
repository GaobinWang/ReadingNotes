# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:15:24 2019

@author: Lesile
"""


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
import pandas as pd
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





















