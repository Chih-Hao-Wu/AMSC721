import numpy as np
import sympy

def define_step(vars: str, functions: list) -> tuple:
    # collect symbols
    sympy_vars = [sympy.symbols(vars)]

    # write functions
    sympify_function = [sympy.sympify(f) for f in functions]
    F = sympy.Matrix(sympify_function)

    # create jacobian
    J = F.jacobian(sympy_vars)

    # return jacobian and functions
    return (sympy_vars[0], F, J)

def iterative_step(vars: tuple, vals: tuple, \
    functions: sympy.matrices.dense.MutableDenseMatrix, \
    jacobian: sympy.matrices.dense.MutableDenseMatrix) -> tuple:
    # take functions and jacobian, do substitution
    s = dict(zip(vars, vals))
    f_s, j_s = functions.subs(s), jacobian.subs(s)

    # set up systems of linear equations as JZ = -f
    a = np.array(j_s).astype(np.float64)
    b = -1 * np.array(f_s).astype(np.float64)

    # solve, return as tuple of values updating xn
    z = tuple(np.linalg.solve(a,b).flatten())

    new = tuple(map(lambda a, b: a+b, vals, z))
    return new

def nr_method(prev_value: int, symbl: str, functions: list, iterations: int) -> tuple:
    """
    implement Newton-Raphson method

    Solves a system of nonlinear equations through an iterative
    approach. The sequence converges to a solution, if it exists
    """
    symbols, F, J = define_step(symbl, functions)

    # iteratively (alternatively recurse, but there is a lower limit),
    for i in range(iterations):
        prev_value = iterative_step(symbols, prev_value, F, J)

    return prev_value

def main():
    initial_value = (-0.8, 1.2)
    symbols = 'x y'
    functions = ["x**2 + y**2 - 2", "x**2 - y - 0.5*x + 0.1"]
    selected_iterations = 100

    solution = nr_method(initial_value, symbols, functions, selected_iterations)
    print(solution)

    # first, check if the inital guess is a solution

if __name__ == "__main__":
    main()
