
# %% import public modules
import matplotlib.pyplot as plt
import autograd.numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
# %% import custom modules
from wrappers import lha_analysis_wrapper, mc_analysis_wrapper, udr_analysis_wrapper
from test_functions import booth_function, matyas_function, easom_function, ackley_function, rastrigin_function, sphere_function, rosenbrock_function, beale_function, goldstein_price_function, levi13_function, eggholder_function

plt.close("all")

"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

plot_type = '2d'
function_type = 'booth_function'

function, X, Y, Z, xP, zP = eval('{}()'.format(function_type))

run_other_analysis = False
if run_other_analysis:
    udr_stability, *tmp = udr_analysis_wrapper(xP, function, mu_lst=[0.0] * len(xP), sigma_lst = [1.0] * len(xP))
    mc_stability, *tmp = mc_analysis_wrapper(xP, function, mu_lst=[0.0] * len(xP), sigma_lst = [1.0] * len(xP))

lha_stability, hessian, eigvals, eigvecs = lha_analysis_wrapper(xP, function, mu_lst=[0.0] * len(xP), sigma_lst = [1.0] * len(xP))
eigvals = eigvals / np.max(np.abs(eigvals)) * np.sqrt((np.abs(xP[0]) - np.max(np.abs(X)))**2 +(np.abs(xP[1]) - np.max(np.abs(Y)))**2) 


## Plot the function, minimum point, and eigen vectors in either 2d or 3d
if plot_type == '3d':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap = 'viridis', linewidth=0, antialiased=True)
    ax.scatter(xP[0], xP[1], zP, color='salmon')
    arrows = zip(2*[xP[0]], 
                 2*[xP[1]], 
                 [zP, zP], 
                 [eigvecs[0][0], eigvecs[1][0]],
                 [eigvecs[0][1], eigvecs[1][1]], 
                 [zP, zP],
                 list(eigvals))
    for i, (tipX, tipY, tipZ, tailX, tailY, tailZ, length) in enumerate(arrows):
        ax.quiver(tipX, tipY, tipZ, tailX, tailY, tailZ, length=length, normalize=False)
        
        
        
elif plot_type == '2d':
    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='viridis')
    fig.colorbar(pcm, ax=ax, extend='max')
    
    ax.scatter(xP[0], xP[1], color='salmon')
    arrows = zip(2*[xP[0]], 
                 2*[xP[1]], 
                 [eigvecs[0][0], eigvecs[1][0]],
                 [eigvecs[0][1], eigvecs[1][1]], 
                 list(eigvals))
    for i, (tipX, tipY, tailX, tailY, length) in enumerate(arrows):
        ax.quiver(tipX, tipY, tailX, tailY, scale=length)

plt.title('{}'.format(function_type))
plt.show()
