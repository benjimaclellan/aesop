import autograd.numpy as np

"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

def booth_function():
    def function(x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
    N = 1000
    x, y = np.linspace(-10, 10, N), np.linspace(-10,10, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [1.0, 3.0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP


def matyas_function():
    def function(x):
        return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]
    
    N = 1000
    x, y = np.linspace(-10, 10, N), np.linspace(-10,10, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [0.0, 0.0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP

def ackley_function():
    """
    Does not work as the hessian has undefined components at the minima point (1/sqrt(0))
    """
    def function(x):
        return -20*np.exp(-0.25*np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20
    
    N = 1000
    x, y = np.linspace(-5, 5, N), np.linspace(-5, 5, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [0.0, 0.0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP

def easom_function():
    """
    Does not work well
    """
    def function(x):
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2)) + 1
    
    N = 1000
    x, y = np.linspace(-10, 10, N), np.linspace(-6,6, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [np.pi, np.pi]
    zP = function(xP)
    return function, X, Y, Z, xP, zP


def rastrigin_function():
    def function(x):
        A, n = (10, 2)
        return A*n + x[0]**2 - A*np.cos(2*np.pi*x[0]) + x[1]**2 - A*np.cos(2*np.pi*x[1])
    
    N = 1000
    x, y = np.linspace(-5, 5, N), np.linspace(-5,5, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [0, 0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP



def sphere_function():
    def function(x):
        return x[0]**2 + x[1]**2
    
    N = 1000
    x, y = np.linspace(-5, 5, N), np.linspace(-5,5, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [0, 0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP



def rosenbrock_function():
    def function(x):
        return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
    
    N = 1000
    x, y = np.linspace(-5, 5, N), np.linspace(-5,5, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [1.0, 1.0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP


def beale_function():
    def function(x):
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    
    N = 1000
    x, y = np.linspace(-4.5, 4.5, N), np.linspace(-4.5, 4.5, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [3.0, 0.5]
    zP = function(xP)
    return function, X, Y, Z, xP, zP



def goldstein_price_function():
    def function(x):
        x, y, = x[0], x[1]
        return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    
    N = 1000
    x, y = np.linspace(-2, 2, N), np.linspace(-2, 2, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [0.0, -1]
    zP = function(xP)
    return function, X, Y, Z, xP, zP




def levi13_function():
    def function(x):
        x, y, = x[0], x[1]
        pi = np.pi
        return np.sin(3*pi*x)**2 + (x-1)**2 * (1 + np.sin(3*pi*y)**2) + (y-1)**2 * (1+np.sin(2*pi*y)**2)
    
    N = 1000
    x, y = np.linspace(-10, 10, N), np.linspace(-10,10, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [1.0, 1.0]
    zP = function(xP)
    return function, X, Y, Z, xP, zP



def eggholder_function():
    def function(x):
        x, y, = x[0], x[1]
        return -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + (y+47)))) - x*np.sin(np.sqrt(np.abs(x-(y+47)))) + 959.6407
    
    N = 1000
    x, y = np.linspace(-512, 512, N), np.linspace(-512,512, N)
    
    X, Y = np.meshgrid(x, y)
    XY = [X, Y]
    Z = function(XY)
    
    xP = [512, 404.2319]
    zP = function(xP)
    return function, X, Y, Z, xP, zP