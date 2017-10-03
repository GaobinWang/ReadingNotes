#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:57:48 2017

@author: Lesile
"""

"""
Chapter8 高性能的Python
“不要降低预期去屈就性能，而要提升性能满足预期” 
对于性能关键应用，应该总是检查两件事情：是否使用正确的实现范型；是否使用正确的高性能库。
许多高性能库可以用于加速Python代码的的执行，下面几个库最为常用:
    Cython：用于合并Python和C语言静态编译范型；
    IPython.parallel：用于在本地或者集群上并行执行代码/函数；
    numexpr：用于快速数值计算；
    multiprocessing：Python内建的(本地)并行处理模块；
    Numba：用于为CPU动态编译Python代码；
    NumbaPro：用于为多核CPU和GPU动态编译Python代码。
"""

###用于比较不同代码性能的函数
def perf_comp_data(func_list, data_list, rep=3, number=1):
    """
    用于比较不同函数性能的函数.
    Parameters
    ------
    func_list:list
        list with function names as strings
    data_list:list
        list with data set names as strings
        
    """
    print("HelloWorld!")

###在GPU上生成随机数
"""
有一个金融领域可以从GPU的使用中得到很大好处：蒙特卡洛模拟，特别是随机数的生成。
"""
import numpy as np
from numbapro.cudalib import curand


###numba
import os
import numba
#@numba.jit()
def c(n):
	count=0
	for i in range(n):
		for i in range(n):
			count+=1
	return count

n=99999
c(n)

