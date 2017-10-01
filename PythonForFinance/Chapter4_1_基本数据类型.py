#!/usr/bin/python


"""
拙劣的程序员担心代码。好的程序员担心数据结构及他们的关系。 --Linus Torvalds

本章主要介绍Python的数据类型和数据结构。具体内容如下：
#基本数据类型
#基本数据结构
#Numpy数据结构
#代码向量化
"""

###基本数据类型
a = 10
type(a)
print(type(a))
"""
内建函数type为所有使用标准和内建类型的对象和新创建的类及对象提供类型的信息。
在后一种情况下，提供的信息取决于程序员存储在类中的描述。
Python中一切皆对象。因此，即使简单的int对象也有内建方法，比如bit_length
"""
a.bit_length() #表现int对象所需的位数
a = 100000000
a.bit_length()
"""
一般来说，我们很难记住所有类和对象的所有方法。高级Python环境(比如IPython)提供了Tab键完成功能，
显示所有对象连接的所有方法。也可以使用Python的内建函数dir显示对象属性和方法的完整列表。
"""

"""
Python面向对象
"""
class Ren():
    """
    定义一个人，给出人的一些属性和方法
    """
    name = "Job"
    hight = "178cm"
    weight = "70kg"
    def run(self,n):
        for i in range(n):
            print("walk step:",str(i))
    def sing(self,n):
        for i in range(n):
            print("sing step:",str(i))
a = Ren()
a.hight
a.name
a.weight
a.run(10)
a.sing(10)
type(a)
help(a)
dir(a)

#Python中的整数可以任意大(解释程序简单的使用位数来表现数值)
googol = 10 ** 100
print(googol)
googol.bit_length

a = 10 ** 100 + 1
b = 10 ** 100 + 2
c = a + b
print(c)
#Python的int对象进行数学运算后返回的仍然是int对象(Python3中下面两种方法已无区别)
1/4
type(1/4)

1.0/4
type(1.0/4)


###浮点数
"""
浮点数与通常不精确的实数计算机表现形式关系更大,取决于所采用的具体技术方法。
"""
b = 0.35
b.is_integer()
b.as_integer_ratio()
dir(b)
#这种浮点对象在内部总是只表现为某种精度
c = b + 0.1
"""
python中浮点数在内部以二进制的形式表示；也就是说，十进制数n(0<n<1)表现为如下形式:
    n = x/2 + y/4 + z/8 +...
    二进制表象形式可能包含大量元素甚至一个无限系列。
但是考虑到用于表达这种数值的位数是固定的--也就是表现系列中的项目固定--结果是不精确的。
"""
c = 0.5
c.as_integer_ratio()

"""
decimal模块提供了一个任意精度浮点数对象.在金融中，确保高精度、超出64位双精度标准有时是必要的
"""
import decimal
from decimal import Decimal
decimal.getcontext()
d = Decimal(1) / Decimal(11)
d
#可以改变Context各个属性的值，从而改变其精度
decimal.getcontext().prec = 4
e = Decimal(1) / Decimal(11)
e

decimal.getcontext().prec = 50
f = Decimal(1) / Decimal(11)
f
#不同精度的数值计算，取最高精度
g = d + e + f  
g


###字符串
t = "this is a string object"
t.capitalize() #将第一个词改为首字母大写
t.split() #将字符串拆分成单个单词，以获取包含所有单词的列表对象
t.find('string') #搜索某个词以得到第一个字母的位置(即索引值)
t.find('Python') #不存在的话返回-1
t.replace(' ','|') #替换
t.upper() #全部字母变为大写
t.count('a') #计算子字符串出现的次数
'http://www.python.org'.strip('http:/') #剥离


"""
#正则表达式
解析字符串对象时，考虑使用正则表达式，可以为这些操作带来便利和高性能
使用字符串对象时，正则表达式是一个很强大的工具
"""
import re
series = """
'01/18/2014 13:00:00', 100, '1st';
'01/18/2014 13:30:00', 120, '2st';
'01/18/2014 14:00:00', 130, '3st'
"""
#下面的正则表达式描述了上述字符串对象中提供的日期-时间信息格式
dt = re.compile("'[0-9/:\s]+'")
result = dt.findall(series)
#对上述字符串格式的日期可以进行解析，生成python日期时间对象
from datetime import datetime
pydt = datetime.strptime(result[0].replace("'",""),'%m/%d/%Y %H:%M:%S')
pydt
type(pydt)