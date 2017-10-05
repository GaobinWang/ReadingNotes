#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:48:03 2017

@author: Lesile
"""

"""
###Chapter7 输入/输出操作
1,输入/输出(I/O)操作通常是金融应用和数据密集型应用当中非常重要的任务。它们往往代表着性能关键计算的瓶颈，因为I/O操作一般无法
以足够快的速度将数据写入RAM和从RAM写入磁盘。在某种意义上,CPU常常因为缓慢的I/O操作而"挨饿"。
2,我们的测算和最近的其他工作表明,现实世界中大部分分析工作处理的输入小于100GB，但是Hadoop/MapReduce等流行基础构架最初是
为PB级处理设计的。 --Appuswamy等人(2013)
3,单一金融分析任务通常处理的数据不超过几个GB--这是Python及其科学栈中的库(如NumPy、pandas和PyTables)最有效的区域。
这样大小的数据集也可以在内存中分析，利用现在的CPU和GPU通常可以获得很高的速度。然而数据必须读入RAM、结果要写入磁盘，
同时要确保满足当今的性能要求。

本章介绍如下领域:
    基本I/O
    pandas的I/O
    PyTables的I/O
"""

"""
###7.1 Python基本I/O
"""

"""
###7.1.1 将对象写入磁盘
pickle模块可以序列化大部分Python对象。
序列化指的是将对象(层次结构)转化为一个字节流；反序列化是相反的操作。
pickle.dump:将对象写入到磁盘
pickle.load:将对象加载到内存
"""
import numpy as np
from random import gauss
import pickle

path = "E:\\Github\\ReadingNotes\\PythonForFinance\\"

#产生随机数
n = 1000000 #100万个随机数
a = [gauss(1.5,2) for i in range(n)]
type(a)
b = np.array(a)
b.nbytes

#打开需要写入的文件
pkl_file = open(path + 'data.pkl','wb') #open file for writing . existing file might be overwritten

#将文件从内存写出到磁盘
%time pickle.dump(a,pkl_file)
pkl_file
pkl_file.close()

#将文件从磁盘读入到内存
pkl_file = open(path + 'data.pkl','rb') #open file for reading
%time b = pickle.load(pkl_file)

#数据比对
a[:5]
b[:5]
np.allclose(np.array(a),np.array(b))
np.sum(np.array(a) - np.array(b))

##对多个文件从内存写出到磁盘
pkl_file = open(path + 'data.pkl', 'wb')  # open file for writing
%time pickle.dump(np.array(a), pkl_file)
%time pickle.dump(np.array(a) ** 2, pkl_file)
pkl_file.close()

#将多个文件从磁盘读入到内存
pkl_file = open(path + 'data.pkl', 'rb')  # open file for reading
x = pickle.load(pkl_file)
x

y = pickle.load(pkl_file)
y
pkl_file.close()
np.allclose(x ** 2,y)

"""
pickle按照先进先出(FIFO)原则保存对象。这种方法有一个重大问题：没有任何可用的元信息，让用户事先知道保存在pickle
文件中的是什么。有时候可以采取一种变通的方法：不存储单个对象，而是存储包含所有其他对象的字典对象。
"""
pkl_file = open(path + 'data.pkl', 'wb')  # open file for writing
pickle.dump({'x' : x, 'y' : y}, pkl_file)
pkl_file.close()

#可以在字典对象的关键值上循环
pkl_file = open(path + 'data.pkl', 'rb')  # open file for writing
data = pickle.load(pkl_file)
pkl_file.close()
for key in data.keys():
    print(key, data[key][:4])
#该方法需要我们一次写入和读取所有对象，在许多情况下，人们可能需要为了更便利而承受这个问题。

"""
###7.1.2 读写文本文件
step1:利用np.random.standard_normal和pd.data_range函数生成数据
step2:利用open函数和write函数将数据逐行写入文件
step3:利用readline或者readlines函数从磁盘读入内存
"""
import numpy as np
import pandas as pd

#step1 生成数据
rows = 50000
cols = 5
data = np.random.standard_normal((rows,cols))
datetime = pd.date_range(start = '2000-01-01',periods = rows,freq = 'H')

#将数据写入到磁盘
csv_file = open(path + 'data.csv','w') #file for writing
header = "datetime,col1,col2,col3,col4,col5\n"
csv_file.write(header)
for i in range(rows):
    line = "%s,%f,%f,%f,%f,%f\n" % (datetime[i],data[i][0],data[i][1],data[i][2],data[i][3],data[i][4])
    csv_file.write(line)
    print(line)
"""
#另外一种写法
for t_,(no1,no2,no3,no4,no5) in zip(datetime,data):
    line = "%s,%f,%f,%f,%f,%f\n" % (t_,no1,no2,no3,no4,no5)
    csv_file.write(line)
"""
csv_file.close()

#将数据读进内存
csv_file = open(path + 'data.csv','r')
#逐行读入
for i in range(100):
    print(csv_file.readline())
#一次性读入所有内容
content = csv_file.readlines()

csv_file.close()


"""
###7.1.3 SQL数据库
Python可以使用任何类型的SQL数据库,通常也可以使用NoSQL数据库。
Python默认自带SQLite3数据库。
利用这个数据库很容易说明Python处理SQL数据库的方式

SQL数据库是相当广泛的主题；本章无法覆盖其广度和复杂度，只能提供一些基本信息:
    Python和几乎所有数据库技术都能很好的集成；
    基本SQL语法主要由使用的数据库决定；剩下的都是真正的Python风格。
"""
import sqlite3 as sq3

#在数据库中创建一张表格
query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'
con = sq3.connect(path + 'numbs.db')
con.execute(query)
con.commit()

#生成数据并写入数据库中
import datetime as dt
con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
            (dt.datetime.now(), 0.12, 7.3))
data = np.random.standard_normal((10000, 2)).round(5)
for row in data:
    con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
                (dt.datetime.now(), row[0], row[1]))
con.commit()

#从数据库中读取数据
#方法1：读取多行
con.execute('SELECT * FROM numbs').fetchmany(10)
#方法2：逐行读取
pointer = con.execute('SELECT * FROM numbs')
for i in range(3):
    print(pointer.fetchone())

con.close()

"""
###7.1.4 读写Numpy数组(Writing and Reading Numpy Arrays)
Numpy本身有以便利、高性能的方式写入和读取ndarray对象的函数。
用于存储:numpy.save
用于读取:numpy.load

在任何情况下都可以预期，这种形式的数据存储和检索远快于SQL数据库或者使用任何标准pickle库的序列化。
当然这种方法没有SQL数据库的功能性，后面介绍的PyTables对此有所帮助。
"""
##例子1:60M的数据写入和读取时间不超过1s
import numpy as np
#生成数据
dtimes = np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00',
                  dtype='datetime64[m]')  # minute intervals
len(dtimes)
dty = np.dtype([('Date', 'datetime64[m]'), ('No1', 'f'), ('No2', 'f')])
data = np.zeros(len(dtimes), dtype=dty)
data['Date'] = dtimes
a = np.random.standard_normal((len(dtimes), 2)).round(5)
data['No1'] = a[:, 0]
data['No2'] = a[:, 1]
data.nbytes

#写入和读取
%time np.save(path + 'array', data)  # suffix .npy is added
%time np.load(path + 'array.npy')

##例子2：480M的数据读入和写出用时不超过6s
data = np.random.standard_normal((10000, 6000))
%time  np.save(path + 'array', data)
%time np.load(path + 'array.npy')

##例子3：480M的数据读入和写出用时不超过3s(在固态硬盘中)
import os 
path = os.getcwd() + '\\'
data = np.random.standard_normal((10000, 6000))
%time  np.save(path + 'array', data)
%time np.load(path + 'array.npy')
#本电脑的固态硬盘和机械硬盘读取速度都是600M/s

###例子4：480M的数据以DataFrame存储之后，变成了1.09G的数据，写入到固态硬盘中用时2min 40s,读进内存用了16s
data = pd.DataFrame(data)
%time data.to_csv("a.csv")
%time data2 = pd.read_csv("a.csv")


"""
###7.2 Pandas的I/O
pandas库的主要优势之一是可以原生读取和写入不同数据格式，包括:
    csv: read_csv to_csv
    XLS/XLSX: read_excel to_excel
    SQL: read_sql to_sql
    Json: read_json to_json
    HTML: read_html to_html
    HDF: read_hdf to_hdf
"""
#用于比较的数据集,约为38M
import numpy as np
import pandas as pd
data = np.random.standard_normal((1000000, 5)).round(5)
path = "E:\\Github\\ReadingNotes\\PythonForFinance\\"
filename = path + 'numbs'
#我们将再次访问SQLite3,比较使用pandas的替代方法的性能。

"""
###7.2.1 SQL数据库
"""
import sqlite3 as sq3
#创建表格
query = 'CREATE TABLE numbers (No1 real, No2 real,No3 real, No4 real, No5 real)'
con = sq3.Connection(filename + '.db')
con.execute(query)

#将100万行的数据写入表格用时9s
%time con.executemany('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data);con.commit()

#将100万行的数据写到内存用时2s
%time temp = con.execute('SELECT * FROM numbers').fetchall()
print(temp[:2])

import sys
sys.getsizeof(temp)
sys.getsizeof(data)

#直接将SQL查询结果读入NumPy ndarray,并绘图
query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
%time res = np.array(con.execute(query).fetchall()).round(3)

res = res[::100]  # every 100th result
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(res[:, 0], res[:, 1], 'ro')
plt.grid(True); plt.xlim(-0.5, 4.5); plt.ylim(-4.5, 0.5)

"""
###7.2.2 从SQL到pandas
用pandas读取整个表或者查询结果通常更为高效。
在可以将整个表读入内存时，分析查询的执行通常比使用基于磁盘的SQL方法快得多。

总结起来可以对50M的模拟数据集得出如下结论:
    写入SQLite3数据花费数秒,而使用pandas使用不到1s
    从SQL数据库读取数据花费数秒,而pandas花费不到1s
"""
#pandas.io.sql子库中包含处理SQL数据库中所保存数据的函数
import pandas.io.sql as pds
%time data = pds.read_sql('SELECT * FROM numbers', con)
data.head()

#数据在内存中可以进行更快的分析
%time  data[(data['No1'] > 0) & (data['No2'] < 0)].head()
#pandas能够控制更为复杂的查询，但是不能替换结构复杂的关系型数结构
%time res = data[['No1', 'No2']][((data['No1'] > 0.5) | (data['No1'] < -0.5)) & ((data['No2'] < -1) | (data['No2'] > 1))]
plt.plot(res.No1, res.No2, 'ro')
plt.grid(True); plt.axis('tight')

"""
正如预期，只要pandas能够复制对应的SQL语句，使用pandas的内存分析能力可以大大加速。
这不仅是pandas的优势，而是pandas与PyTables紧密集成实现的。
"""

#
h5s = pd.HDFStore(filename + '.h5s', 'w')
%time h5s['data'] = data

h5s
h5s.close()

%time h5s = pd.HDFStore(filename + '.h5s', 'r');temp = h5s['data'];h5s.close()

np.allclose(np.array(temp), np.array(data))

"""
###7.2.3 csv文件数据
"""
%time data.to_csv(filename + '.csv')
%time pd.read_csv(filename + '.csv')[['No1', 'No2','No3', 'No4']].hist(bins=20)

"""
###7.2.4 Excel文件数据
"""
%time data[:100000].to_excel(filename + '.xlsx')
%time pd.read_excel(filename + '.xlsx', 'Sheet1').cumsum().plot()

"""
###7.3 PyTables的快速I/O操作
PyTables是Python与HDF5数据库/文件标准的结合.
它专门为优化I/O操作的性能、最大限度地利用可用硬件而设计.该库的导入名为tables.

在内存中分析方面,PyTables与pandas类似，并不是用于替代SQL数据库，而是引入某些功能，进一步弥补不足。
"""
import numpy as np
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline

"""
###7.3.1 使用表 Working with Tables
"""
#PyTables提供基于文件的数据库格式
filename = path + 'tab.h5'
h5 = tb.open_file(filename, 'w') 

#为了举例，我们生成一个200万行的数据表
rows = 2000000
row_des = {
    'Date': tb.StringCol(26, pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
    }
#创建表格时，我们选择无压缩表格
filters = tb.Filters(complevel=0)  # no compression
tab = h5.create_table('/', 'ints_floats', row_des,
                      title='Integers and Floats',
                      expectedrows=rows, filters=filters)

tab
pointer = tab.row

#生成样本数据
ran_int = np.random.randint(0, 10000, size=(rows, 2))
ran_flo = np.random.standard_normal((rows, 2)).round(5)

#将样本数据集逐行写入表格
for i in range(rows):
    pointer['Date'] = dt.datetime.now()
    pointer['No1'] = ran_int[i, 0]
    pointer['No2'] = ran_int[i, 1] 
    pointer['No3'] = ran_flo[i, 0]
    pointer['No4'] = ran_flo[i, 1] 
    pointer.append()
      # this appends the data and
      # moves the pointer one row forward
tab.flush()

#查看写入数据后的表格
tab

##使用Numpy结构数组，可以更高性能、更Python风格的方式实现相同的结果
dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                                 ('No3', '<f8'), ('No4', '<f8')])
sarray = np.zeros(len(ran_int), dtype=dty)
get_ipython().run_cell_magic('time', '', "sarray['Date'] = dt.datetime.now()\nsarray['No1'] = ran_int[:, 0]\nsarray['No2'] = ran_int[:, 1]\nsarray['No3'] = ran_flo[:, 0]\nsarray['No4'] = ran_flo[:, 1]")

#下面这种方法比之前的方法整整快了一个数量级
get_ipython().run_cell_magic('time', '', "h5.create_table('/', 'ints_floats_from_array', sarray,title='Integers and Floats',expectedrows=rows, filters=filters)")

#现在可以删除重复的表格因为已经不需要了
h5.remove_node('/', 'ints_floats_from_array')

#表(Table)对象现在切片时的表现与典型的Python和Numpy对象类似
tab[:3]
tab[:4]['No4']
#更方便和重要的是，我们可以对表或者表的子集应用Numpy通用函数
%time np.sum(tab[:]['No3'])
%time np.sum(np.sqrt(tab[:]['No1']))
#至于绘图，表对象的表现也与ndarray类似
plt.hist(tab[:]['No3'], bins=30)
plt.grid(True)
print(len(tab[:]['No3']))
#类似典型SQL的语句查询数据
res = np.array([(row['No3'], row['No4']) for row in
        tab.where('((No3 < -0.5) | (No3 > 0.5)) \
                 & ((No4 < -1) | (No4 > 1))')])[::100]
    
plt.plot(res.T[0], res.T[1], 'ro')
plt.grid(True)

#从语法和性能的角度看，以表(Table)对象的形式使用PyTables中保存的数据都像使用Numpy在内存中工作一样
values = tab.cols.No3[:]
print("Max %18.3f" % values.max())
print("Ave %18.3f" % values.mean())
print("Min %18.3f" % values.min())
print("Std %18.3f" % values.std())

results = [(row['No1'], row['No2']) for row in
           tab.where('((No1 > 9800) | (No1 < 200)) \
                    & ((No2 > 4500) & (No2 < 5500))')]
for res in results[:4]:
    print(res)
    
results = [(row['No1'], row['No2']) for row in
           tab.where('(No1 == 1234) & (No2 > 9776)')]
for res in results:
    print(res)
    
"""
###7.3.2 使用压缩表
使用PyTables的主要优势之一是压缩方法。
使用压缩不仅能节约磁盘空间，还能改善I/O操作性能。

当I/O称为瓶颈而CPU能够快速(解)压缩数据时，使用压缩对速度有正面的净效应。
如果你的电脑的SSD的性能较高，那么可能观察不到压缩的速度优势。
但是，使用压缩几乎没有任何劣势。
"""
filename = path + 'tab.h5c'
h5c = tb.open_file(filename, 'w') 

filters = tb.Filters(complevel=4, complib='blosc')


tabc = h5c.create_table('/', 'ints_floats', sarray,
                        title='Integers and Floats',
                      expectedrows=rows, filters=filters)
res = np.array([(row['No3'], row['No4']) for row in
             tabc.where('((No3 < -0.5) | (No3 > 0.5)) \
                       & ((No4 < -1) | (No4 > 1))')])[::100]
%time arr_non = tab.read()
%time arr_com = tabc.read()
h5c.close()

"""
###7.3.3 使用数组
我们已经看到，Numpy内建ndarray对象的快速写入和读取能力。
PyTables在存储和检索ndarray对象时也相当快速高效。
"""
%time arr_int = h5.create_array('/', 'integers', ran_int)
%time arr_flo = h5.create_array('/', 'floats', ran_flo)
h5
h5.close()

"""
###7.3.4 内存外计算
PyTables支持内存外计算，因此可以实现不适合于内存的基于数组计算
"""
filename = path + 'array.h5'
h5 = tb.open_file(filename, 'w') 

#创建一个EArray对象，它的第一维可以扩展，而第二维固定宽度为1000
n = 1000
ear = h5.create_earray(h5.root, 'ear',
                      atom=tb.Float64Atom(),
                      shape=(0, n))

#因为EArray对象可以扩展，所以可以块的形式填充
get_ipython().run_cell_magic('time', '', 'rand = np.random.standard_normal((n, n))\nfor i in range(750):\n    ear.append(rand)\near.flush()')

#为了从逻辑上和物理上检查生成数据的多少，我们可以检查为对象提供的元信息和磁盘空间消耗
ear
ear.size_on_desk

#EArray对象有6GB，为了进行内存外计算，我们需要在数据库中有一个目标EArray对象
out = h5.create_earray(h5.root, 'out',
                      atom=tb.Float64Atom(),
                      shape=(0, n))
#PyTables有一个特殊模块可以高效地处理数值表达式，这个模块叫做Expr，基于数值表达式库numexpr。

"""
下面的代码展现了内存外计算的能力（Wall time: 2min 50s）
考虑到整个运算在内存外运行，这样的结果应该相当快速了
"""
expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))')
  # the numerical expression as a string object
expr.set_output(out, append_mode=True)
  # target to store results is disk-based array
%time expr.eval()
  # evaluation of the numerical expression
  # and storage of results in disk-based array

out[0, :10]

"""
我们简单讲其与numexpr模块在内存中的性能做个比较
读取数据(Wall time: 1min 54s)
计算直接无法进行，因为我的电脑内存为4G<6G
"""
%time imarray = ear.read()
  # read whole array into memory

import numexpr as ne
expr = '3 * sin(imarray) + sqrt(abs(imarray))'

ne.set_num_threads(16)
%time ne.evaluate(expr)[0, :10]
h5.close()