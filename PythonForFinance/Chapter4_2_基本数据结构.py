###################
#Python数据结构
##################
"""
一般来说，数据结构是包含其他对象（可能很多）的对象。Python提供如下内建结构：
#元组(任意对象的集合,只有少数可用方法)
#列表(任意对象的集合,有许多可用方法)
#字典(键-值存储对象)
#集合(其他独特元素的无序集合对象)
"""
###元组(tuple)
###列表(list)
a = [1,2,"a"]
a.append(5)
dir(a)
###离题:列表推导&函数式编程&匿名函数
"""
#列表推导
python的特殊性能之一是列表推导,这种方式不在现有的列表对象上循环,而是以紧凑的方式通过循环生成列表对象
#列表推导&函数式编程&匿名函数
在Python级别上尽可能避免循环被视为"好的习惯"。列表推导和函数式编程工具(map filter和reduce)提供了
编写紧凑的(一般来说也易于理解)无循环代码的手段
"""
#列表推导
m = [i ** 2 for i in range(10)]
m
#函数式编程
def f(x):
    return x ** 2
f(2)
def even(x):
    return x % 2 == 0
even(3)

a = map(even, range(10))
for i in a:
    print(i)
a = filter(even,range(15))
for i in a:
    print(i)
"""
reduce函数
在Python 3里,reduce()函数已经被从全局名字空间里移除了,它现在被放置在fucntools模块里 用的话要 先引
入：
from functools import reduce 
"""
from functools import reduce 
reduce(lambda x,y:x + y, range(101))
###字典(dict)


###集合(set)
"""
虽然集合论是数学和金融理论的基石，但是集合对象并没有太多实际应用。
集合对象的应用之一是去掉列表对象中的重复
"""
#集合对象中的所有操作
s = set(['u','d','ud','du','d','du'])
t = set(['d','dd','u','uu'])
s.union(t) #并集
s.intersection(t) #交集
s.difference(t) #差集


#去掉列表对象中的重复
from random import randint
l = [randint(0,10) for i in range(1000)] #生成1000个0-10之间的随机整数
len(l)
l[:20]
s = set(l)
s
a = list(s)