# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 08:05:47 2018

@author: Lesile
"""


import numpy as np
import sys

#Numpy数据类型体系
"""
我们可以使用np.issubdtype判断数组中存储的对象是否是整数，浮点数，字符串或Python对象
"""
ints = np.ones(10,dtype=np.uint16)
sys.getsizeof(ints) #查看对象所占的大小
floats = np.ones(10, dtype=np.float32)
sys.getsizeof(floats)

np.issubdtype(ints.dtype,np.integer)
np.issubdtype(floats.dtype,np.float)

np.float64.mro() #调用dtype的mro方法可查看其所有父类

"""
data_size = 10**8
a =  np.ones(data_size, dtype=np.float32)
a1 = np.ones(data_size, dtype=np.uint16)
b = pd.DataFrame(a)
sys.getsizeof(a)/1024**2
sys.getsizeof(a1)/1024**2
sys.getsizeof(b)/1024**2

"""

##高级数据操作
"""
除了花式索引、切片、布尔条件取子集等操作之外，数组的操作方式还有很多。虽然pandas中的高级函数可以处理数据分析工作中的许多重型任务，但有时你还是需要编写一些在现有库中找不到的数据算法。
"""
#数据重塑
arr = np.arange(8)
arr.reshape((4,2))
arr.reshape((4,2)).reshape((2,4))
#作为参数形状的其中一维可以是-1,,他表示该维度的大小由数据本身推断而来:
arr = np.arange(15)
arr.reshape((5,-1))
#由于数组的shape属性是一个元组,因此它也可以被传入reshape
other_arr = np.ones((3,5))
other_arr.shape
arr.reshape(other_arr.shape)
#与reshape将一维数组转化为多维数组的运算过程相反的运算通常被称为扁平化
arr = np.arange(15).reshape((5,3))
arr.ravel() #不产生数据副本
arr.flatten() #产生数据副本

#C和Fortran顺序
"""
Numpy允许你更为灵活地控制数据在内存中的布局。默认情况下，Numpy数组是按行优先顺序创建的。
由于历史原因，行优先和列优先分别被称为C和Fortran顺序。
reshape和reval这样的函数，都可以接受一个表示数据存放顺序的order参数.
一般可以是"C"或"F"
"""
arr = np.arange(12).reshape((3,4))
np.arange(12).reshape((3,4),order = 'F')
arr.ravel()
arr.ravel('F')

"""
二维或更高维数组的重塑过程比较令人费解.C和Fortran顺序的关键区别就是维度的行进顺序：
C/行优先顺序:先经过更高的维度(例如，轴1会先于轴0被处理)
Fortran/列优先顺序:后经过更高的维度(例如，轴0会先于轴1被处理)
"""

#数组的合并和拆分
#numpy.concatenate可以按指定轴将一个数组组成的序列(如元组、列表等)链接到一起
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8,9],[10,11,12]])
np.concatenate([arr1,arr2],axis=0)
np.concatenate([arr1,arr2],axis=1)

#对于常见的操作numpy提供了一些比较简便的方法(如vstack和hstack)
np.vstack((arr1,arr2))
np.hstack((arr1,arr2))

#split用于将一个数组沿指定轴拆分为多个数组
from numpy.random import randn
arr = randn(5,2)
first,second,third = np.split(arr,[1,3]) #arr[:1] arr[1:3] arr[3:]
#numpy命名空间有两个特殊的对象r_和c_，他们可以使数组的堆叠操作更为简洁

#元素的重复操作:tile和repeat
"""
与其他流行的数据编程语言不通，Numpy中很少需要对数组进行重复操作。这主要是因为广播能更好地满足需求
"""
#repeat会将数组中的各个元素重复一定的次数，从而产生一个更大的数组
arr = np.arange(3)
arr.repeat(3)
#默认情况下repeat传入的数据是一个整数,当然也可以传入一定的数组
arr.repeat([2,3,4])
#对于多维数组，还可以让他们的元素沿着指定轴重复
#如果没有设置轴向，则数据会被扁平化，这可能不是我们想要的结果
arr = randn(2,2)
arr
arr.repeat(2,axis=0)
#同样对于多维数组也可以传入一组整数
arr.repeat([2,3],axis=0)
arr.repeat([2,3],axis=1)

#tile的功能是沿指定的轴堆叠数组的副本
np.tile(arr,2)
np.tile(arr,(2,1))
np.tile(arr,(3,2))

###广播
"""
广播指的是不同形状的数组之间的算术运算的执行方式。
它是一种非常强大的功能，但也很容易令人误解，即使是经验丰富的老手也是如此。
将标量值跟数组合并时就会发生最简单的广播
"""
arr = np.arange(5)
arr*4 #这里我们说:在这个乘法运算中，标量值4被广播到其他所有元素上


###ufunc高级应用
"""
通用函数(即ufunc)是一种对ndarray中的数据执行元素级运算的函数。你可以将其看成简单函数的矢量化包装器。
虽然许多Numpy用户只会用到通用函数所提供的快速元素级运算，但通用函数实际上还有一些高级用法能使我们丢开循环而编写出更为简洁的代码
"""
#reduce接受一个数组参数，并通过一系列的二元运算对其进行聚合(可指明轴向)
#reduce 通过连续执行原始运算的方式进行聚合
#accumulate 聚合值，保留所有局部聚合结果

arr = np.arange(10)
np.add.reduce(arr) #起始值取决于ufunc(对于add的情况就是0)
arr.sum()


#accumulate跟reduce的关系就像是cumsum和sum的关系一样
arr = np.arange(15).reshape((3,5))
np.add.accumulate(arr,axis=1)
#outer用于计算两个数组的叉积
#outer(x,y) 对x和y的每对元素应用原始运算。结果数组的形状为x.shape+y.shape
arr = np.arange(3).repeat([1,2,3])
arr
np.multiply.outer(arr,np.arange(5))
np.subtract.outer(randn(3,4),randn(5))

#reduceat用于计算“局部简约”,其实就是一个对数据各切片进行聚合操作的groupby运算。但其灵活性不如pandas的groupby功能
arr = np.arange(10)
np.add.reduceat(arr,[0,5,8])
#自定义ufunc

###结构化与记录式数组


###高级数组输入输出
"""
np.save和np.load可以用于读写磁盘上以二进制格式存储的数组。
其实还有一些工具可以用于更为复杂的场景。
尤其是内存映像，它使你能处理在内存中放不下的数据集
"""
##内存映像文件
"""
内存映像文件是一种将磁盘上非常大的二进制数据文件当做内存中的数据进行处理的方式。
"""
#使用np.memmap并传入一个文件路径、数据类型、形状以及文件模式，即可创建一个新的memmap:
mmap = np.memmap('mymmap',dtype='float64',mode='w+',shape=(10000,10000))
mmap
#对memmap切片会返回磁盘上数据的视图
section = mmap[:5]
#如果将数据赋值给这些视图:数据会先被缓存在内存中,调用flush即可将其写入磁盘
section[:] = np.random.randn(5,10000)
mmap.flush

del mmap

#当打开一个已经存在的内存映像时，仍然需要指明数据类型和形状，因为磁盘上的那个文件只是一块二进制数据而已，没有任何原数据
mmap = np.memmap('mymmap',dtype='float64',shape=(10000,10000))


###HDF5及其他数组存储方式
"""
PyTables和h5py这两个python项目可以将NumPy的数组数据存储为高效且可压缩的HDF5格式(HDF意思是"层次化数据格式")。
你可以安全地将好几百GB甚至TB的数据存储为HDF5格式
PyTables提供了一些用于结构化数组的高级查询功能，而且还能添加列索引以提升查询速度。
这跟关系型数据库所提供的表索引功能非常类似。
"""

###连续内存的重要性
"""
在某些应用场景中，数组的内存布局可以对计算速度造成极大的影响。
这是因为性能差别在跟CPU的高速缓存(cache)体系有关。运算过程中访问连续内存块(例如，对以C顺序存储的数组进行求和)一般是最快的，
因为内存子系统会将适当的内存块缓存到超高速的L1或L2 CPU Cache中。
此外，Numpy的C语言基础代码(某些)对连续存储的情况进行了优化处理，这样就避免了一些跨越式的内存访问。
一个数组的内存布局是连续的，就是说元素是以他们在数组中出现的顺序存储的内存中的。默认情况下，Numpy数组是以C型连续的方式创建的。
"""
###?在我的电脑上并没有显示出c存储的优越性
arr_c = np.ones((10000,10000),order='C')
arr_f = np.ones((10000,10000),order='F')
arr_c.flags
arr_f.flags
#理论上来说，求和时，arr_c会比arr_f快，因为arr_c的行在内存中是连续的
%timeit arr_c.sum(1)
%timeit arr_f.sum(1)
#如果想从Numpy中提升性能，这里就应该是下手的地方。如果数据的内存顺序不符合你的要求，使用copy并传入'C'或'F'即可解决该问题:
arr_f2 = arr_f.copy('C')
arr_f2.flags
%timeit arr_f2.sum(1)
#注意，在构造视图时，其结果不一定是连续的
arr_c[:50].flags.contiguous

arr_c[:,:50].flags.contiguous


###性能建议
"""
使用Numpy的代码的性能一般都很不错，因为数组运算一般都比纯Pyhton循环快得多。下面大致列出一些需要注意的事项:
    第一，将Python循环和条件逻辑转化为数组运算和布尔数组运算；
    第二，尽量使用广播
    第三，避免复制数据，尽量使用数组视图(即切片)
    第四，使用ufunc及其各种方法.
如果简单Numpy无论如何都达不到所需的性能指标，就可以考虑一下用C Fortran 或Cython来编写代码。
作者自己的工作中经常会用到Cython，因为不需要花费太多精力就可以得到C语言那样的性能。
"""

###其他加速手段：Cython f2py C
"""
近年来，Cython项目已经受到许多python程序员的认可，用它实现的代码运行速度很快(可能需要与C或C++交互,但无需编写纯粹的C代码)。
你可以将Cython看成是带有静态类型并能嵌入C语言的Python。下面这个简单的Cython函数用于对一个一维数组的所有元素求和：
"""  






























