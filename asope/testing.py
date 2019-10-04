import warnings
warnings.filterwarnings("ignore")

# %% import public modules
import matplotlib.pyplot as plt
import autograd.numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# %% import custom modules
from analysis.wrappers import lha_analysis_wrapper, mc_analysis_wrapper, udr_analysis_wrapper
from analysis.test_functions import booth_function, matyas_function, easom_function, ackley_function
from analysis.test_functions import rastrigin_function, sphere_function, rosenbrock_function, beale_function
from analysis.test_functions import goldstein_price_function, levi13_function, eggholder_function,gaussian_function

plt.close("all")

"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
if __name__ == '__main__':

    plot_type = '2d'
    function_type = 'booth_function'

    function, X, Y, Z, xP, zP = eval('{}()'.format(function_type))

    mu_lst, sigma_lst = [0.0] * len(xP), [0.1] * len(xP)
    lha_stability, hessian, eigvals, eigvecs = lha_analysis_wrapper(xP, function, mu_lst=mu_lst, sigma_lst=sigma_lst)
    eigvals = eigvals / np.max(np.abs(eigvals))

    run_other_analysis = True
    if run_other_analysis:
        udr_stability, *tmp = udr_analysis_wrapper(xP, function, mu_lst=mu_lst, sigma_lst=sigma_lst)
        mc_stability, *tmp = mc_analysis_wrapper(xP, function, mu_lst=mu_lst, sigma_lst=sigma_lst, N=10 ** 4)

    xvals = np.arange(len(lha_stability))

    plt.figure()
    plt.stem(xvals-0.1, udr_stability, label='UDR', markerfmt='ro', linefmt ='r--')
    plt.stem(xvals-0.0, mc_stability, label='MC', linefmt='g--', markerfmt='go')
    plt.stem(xvals+0.1, lha_stability, label='LHA', linefmt='b--', markerfmt='bo')
    plt.legend()
    plt.title('{}'.format(function_type))
    plt.show()

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
    ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')
    plt.show()
