import sympy
import math
import numpy as np
import matplotlib.pyplot as plt

def initialize(vars, func):

    x, u = sympy.symbols(vars)
    sympy_func = sympy.sympify(func)
    
    return (x,u,sympy_func)

def iterates_fixed_u(arg_vals, x, u, func, iterations, range_u, stepsize_u):

    lyp_ex = []

    substitutes = dict(zip([x, u], arg_vals))

    u_range = np.arange(substitutes[u], substitutes[u]+range_u, stepsize_u)

    for u_sub in u_range:

        print(u_sub)
        substitutes[u] = u_sub

        sum = 0
        func_diff = sympy.diff(func, x)

        for i in range(iterations):
            sum += math.log(abs(func_diff.subs(substitutes)))

            x_update = func.subs(substitutes)
            # print(substitutes[u], substitutes[x], x_update)
            substitutes[x] = x_update

        #return _iterates

        lyp = sum/iterations

        lyp_ex.append(lyp)

    return lyp_ex

vars = "x u"
function = "u*x*(1-x)"
x, u, func = (initialize(vars, function))

initial_cond = (0.51, 2.8)
range_u = 1.2
stepsize_u = 0.001
a = iterates_fixed_u(initial_cond, x, u, func, 4000, range_u, stepsize_u)

uvalues = np.arange(initial_cond[1], initial_cond[1]+range_u, stepsize_u)
lambdas = a

plt.scatter(uvalues, lambdas, s=1, c='black')
plt.show()
