#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:21:17 2017

@author: Lesile
"""

"""
Chapter9 数学工具
"""

"""
###9.1 逼近法(Approximation)
"""
import seaborn as sns; sns.set()
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#示例函数
def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
plt.plot(x, f(x), 'b')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

###9.1.1 Regression
#Monomials as Basis Functions

"""
np.polyfit函数
用法:polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
功能:Least squares polynomial fit.
说明:Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error.
参数:
    deg : int
    Degree of the fitting polynomial
    
"""
#一次多项式拟合：线性回归
reg = np.polyfit(x, f(x), deg=1)
ry = np.polyval(reg, x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#五次多项式拟合
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#七次多项式拟合
reg = np.polyfit(x, f(x), 7)
ry = np.polyval(reg, x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
#比较拟合效果
np.allclose(f(x), ry)
#计算MSE
np.sum((f(x) - ry) ** 2) / len(x)

##单独的基函数
"""
np.linalg.lstsq函数
用法：lstsq(a, b, rcond=-1)
功能:Return the least-squares solution to a linear matrix equation.
介绍:Solves the equation `a x = b` by computing a vector `x` that
     minimizes the Euclidean 2-norm `|| b - a x ||^2`.
参数:
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
返回:
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(), (1,), (K,)} ndarray
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
"""
matrix = np.zeros((3 + 1, len(x)))
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

reg = np.linalg.lstsq(matrix.T, f(x))[0]
reg
ry = np.dot(reg, matrix)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

matrix[3, :] = np.sin(x)
reg = np.linalg.lstsq(matrix.T, f(x))[0]
ry = np.dot(reg, matrix)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

np.allclose(f(x), ry)
np.sum((f(x) - ry) ** 2) / len(x)
reg

##有噪声数据 Noisy Data

xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg, xn)

plt.plot(xn, yn, 'b^', label='f(x)')
plt.plot(xn, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

## 未排序数据 Unsorted Data
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)

print(xu[:10].round(2))
print(yu[:10].round(2))

reg = np.polyfit(xu, yu, 5)
ry = np.polyval(reg, xu)

plt.plot(xu, yu, 'b^', label='f(x)')
plt.plot(xu, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

##多维数据 Multiple Dimensions
def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

Z = fm((X, Y))
x = X.flatten()
y = Y.flatten()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
        linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

import statsmodels.api as sm
model = sm.OLS(fm((x, y)), matrix).fit()
model.summary
model.rsquared
a = model.params
a

def reg_func(a, p):
    x, y = p
    f6 = a[6] * np.sqrt(y)
    f5 = a[5] * np.sin(x)
    f4 = a[4] * y ** 2
    f3 = a[3] * x ** 2
    f2 = a[2] * y
    f1 = a[1] * x
    f0 = a[0] * 1
    return (f6 + f5 + f4 + f3 +
            f2 + f1 + f0)

RZ = reg_func(a, (X, Y))

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
            cmap=mpl.cm.coolwarm, linewidth=0.5,
            antialiased=True)
surf2 = ax.plot_wireframe(X, Y, RZ, rstride=2, cstride=2,
                          label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)

###9.1.2 插值 Interpolation
import scipy.interpolate as spi
"""
scipy.interpolate.splrep函数
用法:splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None, full_output=0, per=0, quiet=1)
功能:Find the B-spline representation of 1-D curve.
介绍:Given the set of data points ``(x[i], y[i])`` determine a smooth spline
    approximation of degree k on the interval ``xb <= x <= xe``.
参数:
    x, y : array_like(注意x，y必须是有序的)
        The data points defining a curve y = f(x).
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as x and y.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the y values have standard-deviation given by the
        vector d, then w should be 1/d. Default is ones(len(x)).
    xb, xe : float, optional
        The interval to fit.  If None, these default to x[0] and x[-1]
        respectively.
    k : int, optional
        The degree of the spline fit. It is recommended to use cubic splines.
        Even values of k should be avoided especially with small s values.
        1 <= k <= 5
    s : 平滑因子(越大约平滑)
    
"""
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
def f(x):
    return np.sin(x) + 0.5 * x

#一次样条拟合
ipo = spi.splrep(x, f(x), k=1)
iy = spi.splev(x, ipo)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, iy, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

np.allclose(f(x), iy)

xd = np.linspace(1.0, 3.0, 50)
iyd = spi.splev(xd, ipo)

plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#三次样条拟合
ipo = spi.splrep(x, f(x), k=3)
iyd = spi.splev(xd, ipo)

plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

np.allclose(f(xd), iyd)
np.sum((f(xd) - iyd) ** 2) / len(xd)

"""
###9.2 凸优化(Convex Optimization)
"""
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

#定义函数
def fm(x,y):
    return(np.sin(x) + 0.05 * x ** 2
           + np.sin(y) + 0.05 * y ** 2)
fm(1,2)
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X,Y = np.meshgrid(x, y)
Z = fm(X,Y)

#3D绘图
fig = plt.figure(figsize = (9,6))
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2,
                       cmap = plt.cm.coolwarm,
                       linewidth = 0.5,
                       antialiased = True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()

import scipy.optimize as spo

###9.1.1 全局优化
#遍历的方法进行优化：brute
"""
scipy.optimize.brute函数
用法:brute(func, ranges, args=(), Ns=20, full_output=0, finish=<function fmin at 0x000002159949C378>, disp=False)
功能:Minimize a function over a given range by brute force.
介绍:Uses the "brute force" method, i.e. computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.
参数:
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be either a
        "slice object" or a range tuple of the form ``(low, high)``.
        The program uses these to create the grid of points on which
        the objective function will be computed. See `Note 2` for
        more detail.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified. See `Note2`.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
"""
def fo(value):
    x, y = value
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f %8.4f %8.4f' % (x, y ,z))
    return(z)
output = True
fo((1,1))
result = spo.brute(fo,((-10, 10.1, 5),(-10, 10.1, 5)),full_output=True,finish = None)
result[0]  # global minimum
result[1]  # function value at global minimum
result = spo.brute(fo,((-10, 10.1, 0.1),(-10, 10.1, 0.1)),full_output=True,finish = None)
result[0]  # global minimum
result[1]  # function value at global minimum
opt1 = result[0]
fo(opt1)


###9.2.2 局部优化
"""
对于局部凸优化,我们打算利用全局优化的结果。

spo.fmin函数
用法:fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, 
        full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)
功能:This algorithm only uses function values, not derivatives or second
    derivatives.
参数:
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func, i.e. ``f(x,*args)``.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : number, optional
        Maximum number of function evaluations to make.
        
"""
output = True
opt2 = spo.fmin(fo, opt1,xtol = 0.001, ftol = 0.001,
                maxiter = 15, maxfun = 20)
opt2
fo(opt2)
"""
在许多凸优化算法中，在进行局部优化之前进行全局优化。
主要原因是局部凸优化算法很容易陷入某个局部最小值，而忽略"更好的"局部最小值和全局最小值。
下面的例子可以看到，将初始参数设置为x = y = 2得出高于0的"最小值":
"""
output = False
result = spo.fmin(fo,(2.0,2.0),maxiter = 250)
fo(result)


###9.2.3 有约束的优化问题
from math import sqrt
#需要优化的目标函数
def Eu(value):
    s,b = value
    z = -1*(0.5 * sqrt(s * 15 + b * 5) + 0.5 * sqrt(s * 5 + b * 12))
    return(z)
#约束条件
cons = ({'type':'ineq','fun':lambda value: 100 - value[0] *10 - value[1] * 10})
bnds = ((0,1000),(0,1000))
result = spo.minimize(Eu,[5,5],method = 'SLSQP',
                      bounds = bnds, constraints = cons)
result

"""
###9.3 积分(Integration)
"""
import scipy.integrate as sci

def f(x):
    return np.sin(x) + 0.5 * x

a = 0.5  # left integral limit
b = 9.5  # right integral limit
x = np.linspace(0, 10)
y = f(x)

from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

# area under the function
# between lower and upper limit
Ix = np.linspace(a, b)
Iy = f(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)

# labels
plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",
         horizontalalignment='center', fontsize=20)

plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')

ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)])

###9.3.1 数值积分(Numerical Integration)
"""
integrate子库包含一组精选的函数，可以计算给定上下限和数学函数下的数值积分。
这些函数的例子包含:
    用于固定高斯求积的fixed_quad
    用于自适应求积的quad
    用于龙贝格积分的romberg
"""
sci.fixed_quad(f, a, b)[0]
sci.quad(f, a, b)[0]
sci.romberg(f, a, b)

"""
还有一些积分函数以输入列表或者包含函数值和输入值的ndarray对象作为输入。
这种函数的例子包括:
    使用梯形法则的trapz
    实现辛普森法则的simps
"""
xi = np.linspace(0.5, 9.5, 25)
sci.trapz(f(xi), xi)
sci.simps(f(xi), xi)
###9.3.2 通过模拟求取积分（Integration by Simulation）

for i in range(1, 20):
    np.random.seed(1000)
    x = np.random.random(i * 10) * (b - a) + a
    print(np.sum(f(x)) / len(x) * (b - a))

"""
###9.4 符号计算(Symbolic Computation)
对于交互式的金融分析，符号计算可能是比非符号计算更搞笑的方法.
"""
#SymPy是Python中用于符号计算的库
import sympy as sy 

###9.4.1 基本指数
x = sy.Symbol('x')
y = sy.Symbol('y')

type(x)
sy.sqrt(x)

3 + sy.sqrt(x) - 4**2

#可以用符号对象定义任何函数，他们不会和Python函数混淆:
f = x ** 2 + 3 + 0.5 * x ** 2 + 3/2
sy.simplify(f)

#SymPy为数学表达式提供了3个基本的渲染器
sy.init_printing(pretty_print=False, use_unicode=False)
print(sy.pretty(f))
print(sy.pretty(sy.sqrt(x) + 0.5))

pi_str = str(sy.N(sy.pi, 400000))
pi_str[:40]
pi_str[-40:]
pi_str.find('111272')

###9.4.2 方程式
"""
SymPy的长处之一是解方程
然而，不管是从数学的角度(解的存在性)还是从算法的角度,都很明显不能保证有解

功能:Algebraically solves equations and systems of equations.
    Currently supported are:
        - polynomial,
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - sytems containing relational expressions.
用法:solve(f, *symbols, **flags)

"""
sy.solve(x ** 2 - 1,dict=True)
sy.solve(x ** 2 - 1 - 3)
sy.solve(x ** 3 + 0.5 * x ** 2 - 1)
sy.solve(x ** 2 + y ** 2)
###9.4.3 积分
a, b = sy.symbols('a b')

print(sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))))
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print(sy.pretty(int_func))
Fb = int_func.subs(x, 9.5).evalf()
Fa = int_func.subs(x, 0.5).evalf()
Fb - Fa  # exact value of integral
int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
print(sy.pretty(int_func_limits))
int_func_limits.subs({a : 0.5, b : 9.5}).evalf()
sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5))
###9.4.4 微分

int_func.diff()
f = (sy.sin(x) + 0.05 * x ** 2
   + sy.sin(y) + 0.05 * y ** 2)
   
del_x = sy.diff(f, x)
del_x

del_y = sy.diff(f, y)
del_y


xo = sy.nsolve(del_x, -1.5)
xo

yo = sy.nsolve(del_y, -1.5)
yo

f.subs({x : xo, y : yo}).evalf() 

xo = sy.nsolve(del_x, 1.5)
xo

yo = sy.nsolve(del_y, 1.5)
yo

f.subs({x : xo, y : yo}).evalf()

