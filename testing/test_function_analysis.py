import warnings

# %% import public modules
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# %% import custom modules
from lib.hessian import hessian
from testing.test_functions import TestFunction


# booth_function, matyas_function, easom_function, ackley_function
# rastrigin_function, sphere_function, rosenbrock_function, beale_function
# goldstein_price_function, levi13_function, eggholder_function,gaussian_function

# %%
"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

plt.close("all")

if __name__ == '__main__':


    plot_types = set(('3d', '2d'))

    test = TestFunction('beale_function')

    # function, X, Y, Z, xP, zP = eval('{}()'.format(function_type))

    h = hessian(test.function)(np.array(test.xP))
    (eigvals, eigvecs) = numpy.linalg.eig(h)
    eigvals = eigvals / np.max(np.abs(eigvals))

    xvals = np.arange(len(test.xP))

    plt.figure()
    plt.stem(xvals+0.1, eigvals, label='LHA', linefmt='b--', markerfmt='bo', use_line_collection=True)
    plt.legend()
    plt.title('{}'.format(test.name))
    plt.show()

    ## Plot the function, minimum point, and eigen vectors in either 2d or 3d
    if '3d' in plot_types:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(test.X, test.Y, test.Z, norm=colors.LogNorm(vmin=test.Z.min(), vmax=test.Z.max()),
                               cmap = 'viridis', linewidth=10, antialiased=False, alpha=0.5)
        ax.scatter(test.xP[0], test.xP[1], test.zP, color='black')
        arrows = zip(2*[test.xP[0]],
                     2*[test.xP[1]],
                     [test.zP, test.zP],
                     [eigvecs[0][0], eigvecs[1][0]],
                     [eigvecs[0][1], eigvecs[1][1]],
                     [test.zP, test.zP],
                     list(eigvals))
        for i, (tipX, tipY, tipZ, tailX, tailY, tailZ, length) in enumerate(arrows):
            ax.quiver(tipX, tipY, tipZ, tailX, tailY, tailZ, length=length, normalize=False)



    if '2d' in plot_types:
        fig, ax = plt.subplots(1, 1)
        pcm = ax.pcolormesh(test.X, test.Y, test.Z,
                            norm=colors.LogNorm(vmin=test.Z.min(), vmax=test.Z.max()), cmap='viridis')
        fig.colorbar(pcm, ax=ax, extend='max')

        ax.scatter(test.xP[0], test.xP[1], color='salmon')
        arrows = zip(2*[test.xP[0]],
                     2*[test.xP[1]],
                     [eigvecs[0][0], eigvecs[1][0]],
                     [eigvecs[0][1], eigvecs[1][1]],
                     list(eigvals))
        for i, (tipX, tipY, tailX, tailY, length) in enumerate(arrows):
            ax.quiver(tipX, tipY, tailX, tailY, scale=True)

    plt.title('{}'.format(test.name))
    ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')
    plt.show()
