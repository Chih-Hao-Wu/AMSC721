import sympy

def fp_method(val: float, function: str, iterations: int):
    # rearrange f(x) = 0 in the form of g(x) = x
    # function parameter has g(x) passed as the argument
    var = sympy.symbols('x')
    func = sympy.sympify(function)

    # iteratively, update the value of x as x_n+1 = g(x_n)
    for i in range(iterations):
        val = func.subs({var:val})
        print(val.evalf())
    return val

def main():
    initial_value = 4
    function = "(2*x + 3)**(1/2)"
    iterations = 5
    print(initial_value)
    print(fp_method(initial_value, function, iterations))

if __name__ == "__main__":
    main()
