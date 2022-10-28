from numpy import *
from numpy.linalg import *

class henon_map:

    def __init__(_, parameters):
        # initialize parameters to function
        _.a, _.b = parameters

    def f(_, values):
        # return values of mapping given x and y
        x, y = values
        result = array([1 - _.a * x ** 2 + y, _.b * x])
        return result

    def M(_, values, vector):
        # iterate vectors according to the jacobian
        # because of jacobian, we don't need y
        x = values[0] 
        jacobian = array([[-2*_.a*x, 1],[_.b, 0]])
        return jacobian @ vector

    def __call__(_, values, vector):
        value_vector = (_.f(values), _.M(values, vector))
        return value_vector

def lyapunov(system, initial_values=None, iterations=10000, iter_skip=200):

    n = len(initial_values)      # dims phase space
    vals = initial_values        # initial values for variables

    vector = eye(n)     # arbitrary orthonormal vectors
    sum = zeros(n)      # sum adding to lyapunov exponents

    for _step in range(iterations):

        vals, displaced_vector = system(vals, vector)
        displaced_vector = qr_factorization(displaced_vector)
        sum = sum + natural_logarithm(displaced_vector)
        vector = normalization(displaced_vector)

    lyapunovs = sum/iterations
    return sort(lyapunovs)[::-1]

def qr_factorization(vector):

    q, r = qr(vector)
    return q @ diag(r.diagonal())

def normalization(vector):

    return apply_along_axis(lambda x: x/norm(x), 0, vector)

def natural_logarithm(vector):

    return apply_along_axis(lambda x: log(norm(x)), 0, vector)

parameters = (1.4,0.3) # Henon's original parameters for a chaotic attractor
initial_conditions=array([0.1,0.3]) # Randomly select initial condition for iterates to plot Henon's map

exp = lyapunov(henon_map(parameters), initial_values=initial_conditions).max()
print(f"The maxmimal Lyapunov exponent is: {exp}")
