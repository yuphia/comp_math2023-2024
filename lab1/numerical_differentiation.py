import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def differentiate_function(f, delta, method, x):
    return method(f, delta, x)

def method1(f, delta, x):
    return (f(x+delta) - f(x))/delta

def method2(f, delta, x):
    return (f(x) - f(x-delta))/delta

def method3(f, delta, x):
    return (f(x+delta) - f(x-delta))/(2*delta)

def method4(f, delta, x):
    return (4/3)*(f(x+delta) - f(x-delta))/(2*delta) - (1/3)*(f(x+2*delta) - f(x-2*delta))/(4*delta)

def method5(f, delta, x):
    return (3/2)*(f(x+delta) - f(x-delta))/(2*delta) - (3/5)*(f(x+2*delta) - f(x-2*delta))/(4*delta) + (1/10)*(f(x+3*delta) - f(x-3*delta))/(6*delta)

deltas_range = np.arange(1, 21, 1)
deltas = list (map ((lambda power: 2/(2**power)), (deltas_range)))
deltas.reverse()

methods = [method1, method2, method3, method4, method5]
x = 0.5

sin_arg_squared = lambda x: math.sin(x**2)
sin_arg_squared_deriv = lambda x: 2*x*math.cos(x**2)


sin_cos = lambda x: math.sin(math.cos(x))
sin_cos_deriv = lambda x: -math.sin(x)*math.cos(math.cos(x))

exp_sin_cos = lambda x: math.exp(sin_cos(x))
exp_sin_cos_deriv = lambda x: math.exp(sin_cos(x))*sin_cos_deriv(x)

plus_3 = lambda x: x+3

def plus_3_deriv(x):
    return 0

ln_plus_3 = lambda x: np.log(plus_3(x))
ln_plus_3_deriv = lambda x: 1/(x+3)

sqrt_plus_3 = lambda x: math.sqrt(plus_3(x))
sqrt_plus_3_deriv = lambda x: 1/(2*math.sqrt(plus_3(x)))



functions = [(sin_arg_squared, sin_arg_squared_deriv), (sin_cos, sin_cos_deriv), (exp_sin_cos, exp_sin_cos_deriv), (plus_3, plus_3_deriv), (ln_plus_3, ln_plus_3_deriv)
             , (sqrt_plus_3, sqrt_plus_3_deriv)]

def get_results_single_function_single_method (function, method, results):
    for i in range(1, 21, 1):
        res = differentiate_function(function[0], deltas[i-1], method, x) - function[-1](x)
        results.append(abs(res))

results = []

for func in functions:
    for method in methods:
        res = []
        get_results_single_function_single_method(func, method, res)
        deltas_np = np.asarray(deltas)
        res_np = np.asarray(res)
        results.append((deltas_np, res_np))
        plt.plot(deltas_np, res_np, marker='o')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        #plt.show()    


