
# %% import public modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import autograd.numpy as np
import pandas as pd
import seaborn
from mpl_toolkits.mplot3d import Axes3D
# %% import custom modules
from assets.fitness_analysis import lha_analysis_wrapper, mc_analysis_wrapper

plt.close("all")



func = lambda x, y: (x + 2*y - 7)**2 + (2*x + y -5)**2

X, Y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
xP, yP, zP = (1, 3, 0)

Z = func(X, Y)


fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, Z,cmap='viridis', linewidth=0, antialiased=True)
ax.scatter(xP, yP, zP)
plt.show()
