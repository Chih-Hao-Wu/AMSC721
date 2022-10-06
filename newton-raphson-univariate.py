#univariate 
import sympy

# quick and unrefined; refer to other for cleaner code
def nr_method(var: str, function: list, val: float, iterations = 5, count_step = 0):

    x = sympy.symbols(var)
    func = sympy.sympify(function)

    func_derivative = sympy.diff(func, x)
    assigned_x_value = {x: val}

    print(round(float(val),5))
    val = val - (func.subs(assigned_x_value)/func_derivative.subs(assigned_x_value))

    count_step += 1
    if count_step < iterations:
        nr_method(var, function, val, iterations, count_step)
    else:
        return 1

var = 'x'
equation = "cot(x) - x"
nr_method(var, equation, 6.4, 10)
