import numpy as np
import multiprocess as mp
import dill

def worker(x):
    return x * x

def parallel(n_cores, input_args):
    with mp.Pool(n_cores) as pool:
        res = pool.map(worker, input_args)
    return res

def save_graph(graph, filename):
    
    return

#%% scaling units in plots
def scale_units(ax, unit='', axes=['x']):
    def scale(data):
        prefixes = {18: r"E",
                    15: r"P",
                    12: r"T",
                    9: r"G",
                    6: r"M",
                    3: r"k",
                    0: r"",
                    -3: r"m",
                    -6: r"$\mu$",
                    -9: r"n",
                    -12: r"p",
                    -15: r"f",
                    -18: r"a",
                    -21: r"z"}

        order = np.log10(max(abs(data)))
        multiplier = 3 * int(np.floor(order / 3))

        prefix = prefixes[multiplier]
        return multiplier, prefix

    def pass_function_handles(ax, line, axis):
        if axis == 'x':
            get_data, set_data, get_label, set_label = [line.get_xdata, line.set_xdata, ax.get_xlabel, ax.set_xlabel]
        elif axis == 'y':
            get_data, set_data, get_label, set_label = [line.get_ydata, line.set_ydata, ax.get_ylabel, ax.set_ylabel]
        elif axis == 'z':
            get_data, set_data, get_label, set_label = [line.get_zdata, line.set_zdata, ax.get_zlabel, ax.set_zlabel]

        return get_data, set_data, get_label, set_label

    if ax.lines:
        for axis in axes:
            line = ax.lines[0]
            get_data, set_data, get_label, set_label = pass_function_handles(ax, line, axis)

            data = get_data()
            multiplier, prefix = scale(data)  # always changes based on first line plotted (could cause issues)

            for line in ax.lines:
                get_data, set_data, get_label, set_label = pass_function_handles(ax, line, axis)
                data = get_data()
                set_data(data / 10 ** multiplier)

            label = get_label()
            set_label(label + ' ({}{})'.format(prefix, unit))
            ax.relim()
            ax.autoscale_view()

        return multiplier, prefix
    else:
        return