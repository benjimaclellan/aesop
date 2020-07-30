import autograd.numpy as np

"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


class TestFunction(object):
    __N = 1000

    def __init__(self, name='booth_function'):
        function, x_range, y_range, xP = eval('{}()'.format(name))

        x, y = np.linspace(x_range[0], x_range[1], self.__N), np.linspace(y_range[0], y_range[1], self.__N)
        X, Y = np.meshgrid(x, y)
        XY = [X, Y]
        Z = function(XY)
        zP = function(xP)

        self.name = name

        self.function = function
        self.X = X
        self.Y = Y
        self.Z = Z

        self.xP = xP
        self.zP = zP

        return


THETA = 0.0 * np.pi
def rotate_coordinates(x):
    xp = [None,None]
    xp[0] = x[0] * np.cos(THETA) - x[1] * np.sin(THETA)
    xp[1] = x[0] * np.sin(THETA) + x[1] * np.cos(THETA)
    return xp

def booth_function():
    def function(x):
        x = rotate_coordinates(x)
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    x_range, y_range = [-10, 10], [-10,10]
    xP = [1.0, 3.0]
    return function, x_range, y_range, xP

def matyas_function():
    def function(x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

    x_range, y_range = [-10, 10], [-10, 10]
    xP = [0.0, 0.0]
    return function, x_range, y_range, xP



def ackley_function():
    """
    Does not work as the hessian has undefined components at the minima point (1/sqrt(0))
    """

    def function(x):
        return -20 * np.exp(-0.25 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.exp(1) + 20

    x_range, y_range = [-5, 5], [-5, 5]
    xP = [0.0, 0.0]
    return function, x_range, y_range, xP


def easom_function():
    """
    Does not work well
    """

    def function(x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)) + 1

    x_range, y_range = [-10, 10], [-6, 6]
    xP = [np.pi, np.pi]
    return function, x_range, y_range, xP



def rastrigin_function():
    def function(x):
        A, n = (10, 2)
        return A * n + x[0] ** 2 - A * np.cos(2 * np.pi * x[0]) + x[1] ** 2 - A * np.cos(2 * np.pi * x[1])

    x_range, y_range = [-5, 5], [-5, 5]
    xP = [0.0, 0.0]
    return function, x_range, y_range, xP


def sphere_function():
    def function(x):
        return x[0] ** 2 + x[1] ** 2

    x_range, y_range = [-5, 5], [-5, 5]
    xP = [0.1, 0.1]
    return function, x_range, y_range, xP


def rosenbrock_function():
    def function(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x_range, y_range = [-5, 5], [-5, 5]
    xP = [1.0, 1.0]
    return function, x_range, y_range, xP



def beale_function():
    def function(x):
        return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
                    2.625 - x[0] + x[0] * x[1] ** 3) ** 2

    x_range, y_range = [-4.5, 4.5], [-4.5, 4,5]
    xP = [3.0, 0.5]
    return function, x_range, y_range, xP


def goldstein_price_function():
    def function(x):
        x, y, = x[0], x[1]
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    x_range, y_range = [-2, 2], [-2, 2]
    xP = [0.0, -1.0]
    return function, x_range, y_range, xP


def levi13_function():
    def function(x):
        x, y, = x[0], x[1]
        pi = np.pi
        return np.sin(3 * pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * pi * y) ** 2) + (y - 1) ** 2 * (
                    1 + np.sin(2 * pi * y) ** 2)

    x_range, y_range = [-10, 10], [-10, 10]
    xP = [1.0, 1.0]
    return function, x_range, y_range, xP


def eggholder_function():
    def function(x):
        x, y, = x[0], x[1]
        return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
            np.sqrt(np.abs(x - (y + 47)))) + 959.6407

    x_range, y_range = [-512, 512], [-512, 512]
    xP = [512.0, 404.2319]
    return function, x_range, y_range,

def gaussian_function():
    def function(x):
        return np.exp(-0.5 * (1 * x[0] ** 2 + 50 * x[1] ** 2))
    x_range, y_range = [-5, 5], [-5, 5]
    xP = [0.0, 0.0]
    return function, x_range, y_range,
