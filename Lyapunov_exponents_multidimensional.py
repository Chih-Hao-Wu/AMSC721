from scipy import integrate
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt

class HelperFunctions:
    """helping methods for matrix operations, packing and unpacking values"""

    def pack(values, vector):
        matrix = concatenate((values, reshape(vector, 9)), axis=0)
        return matrix

    def unpack(item):
        return item[0:3], reshape(item[3::], (3, 3))

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

class LorenzEquations:

    def __init__(_, parameters, time_step = 0.01):
        # initialize parameters to function
        _.s, _.r, _.b = parameters
        _.h = time_step # time step


    def variational_equation(_, item, t=None):
        values, vector = HelperFunctions.unpack(item)
        x, y, z = values

        dot_values = array([_.s * (-x + y),
                            x * (_.r - z) - y,
                            x * y - _.b * z])

        dot_vectors = array([[ -_.s, _.s, 0],
                            [_.r - z, -1, -x],
                            [y, x, -_.b]]) @ vector

        return HelperFunctions.pack(dot_values, dot_vectors)

    def __call__(_, values, vector):
        item = HelperFunctions.pack(values, vector)
        next_item = integrate.odeint(_.variational_equation, item, array([0, 1]), h0=_.h)
        return HelperFunctions.unpack(next_item[1])

class RosslerSystem:

    def __init__(_, parameters, time_step=0.10):
        _.a, _.b, _.c = parameters
        _.h = time_step

    def variational_equation(_, item, t=None):
        values, vector = HelperFunctions.unpack(item)
        x, y, z = values

        dot_values = array([-1 * (y + z),
                         x + _.a * y,
                         _.b + z * (x - _.c)])

        dot_vectors = array([[ 0, -1, -1],
                       [1, _.a, 0],
                       [z,  0, (x - _.c)]]) @ vector

        return HelperFunctions.pack(dot_values, dot_vectors)

    def __call__(_, values, vector):
        item = HelperFunctions.pack(values, vector)
        next_item = integrate.odeint(_.variational_equation, item, array([0, 1]), h0=_.h)
        return HelperFunctions.unpack(next_item[1])


def lyapunov(system, initial_values=None, iterations=2000):

    n = len(initial_values)      # dims phase space
    vals = initial_values        # initial values for variables

    vector = eye(n)     # arbitrary orthonormal vectors
    sum = zeros(n)      # sum adding to lyapunov exponents

    for _step in range(iterations):

        vals, displaced_vector = system(vals, vector)
        displaced_vector = orthogonalization(displaced_vector)
        sum = sum + natural_logarithm(displaced_vector)
        vector = normalization(displaced_vector)

    lyapunovs = sum/iterations
    return sort(lyapunovs)[::-1]

def orthogonalization(vector):
    q, r = qr(vector)
    diagonal_values = diag(r.diagonal())
    return q @ diagonal_values 

def normalization(vector):
    return apply_along_axis(lambda x: x/norm(x), 0, vector)

def natural_logarithm(vector):
    return apply_along_axis(lambda x: log(norm(x)), 0, vector)

parameters = (1.4,0.3) # Henon's original parameters for a chaotic attractor
initial_values=array([0.1,0.3]) # Randomly select initial condition for iterates to plot Henon's map
exp = lyapunov(henon_map(parameters), initial_values=initial_values).max()
print(f"The maxmimal Lyapunov exponent is: {exp}")

parameters = (10, 28, 8/3)
initial_values = array([1,1,1])
exp = lyapunov(LorenzEquations(parameters), initial_values=initial_values).max()
print(f"The maxmimal Lyapunov exponent is: {exp}")

"""
parameters = (0.15, 0.20, 10)
initial_values = array([0,1,2])
exp = lyapunov(RosslerSystem(parameters), initial_values=initial_values).max()
print(f"The maxmimal Lyapunov exponent is: {exp}")
"""
