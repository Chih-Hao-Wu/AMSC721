# consider a polynomial function f(x) = x^2 - 2x - 3 = 0
# rearrange into the form of g(x) = x = sqrt(2x+3)

import math

# solve as a fixed-point problem using recursion
def fp_problem(n_value: int, iterations: int, count=0) -> float:

    # print(n_value)
    n_value = func(n_value)
    count += 1

    if count < iterations:
        return fp_problem(n_value, iterations, count)
    else:
        return n_value

def func(x: float) -> float:
    # function
    return math.sqrt(2*x+3)

# upper limit of recursion depth is 997
print(fp_problem(4, 997))
