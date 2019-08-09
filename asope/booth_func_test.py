
# %% import public modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import autograd.numpy as np
import pandas as pd
import seaborn
from mpl_toolkits.mplot3d import Axes3D
# %% import custom modules
from analysis.wrappers import lha_analysis_wrapper, mc_analysis_wrapper

plt.close("all")



#def func(x):
#    return np.power(x[0] + 2*x[1] - 7,2) + np.power(2*x[0] + x[1] - 5, 2)

def wrapper(X):
    return func(X[0], X[1])
def func(x, y):
    return np.power(x + 2*y - 7, 2) + np.power(2*x + y - 5, 2)

X, Y = np.meshgrid(np.linspace(-10,10,100).astype('double'), np.linspace(-10,10,100).astype('double'))
XY = [X, Y]

xP, zP = ([1.0, 3.0], 0)


Z = func(X, Y)

lha_analysis_wrapper(xP, wrapper, [0,0], [1,1])

#fig = plt.figure()
#ax = fig.gca(projection='3d')
## Plot the surface.
#surf = ax.plot_surface(X, Y, Z,cmap='viridis', linewidth=0, antialiased=True)
##ax.scatter(xP, yP, zP, )
#plt.show()
