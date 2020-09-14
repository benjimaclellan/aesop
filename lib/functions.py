import numpy as np
import multiprocess as mp
import dill
import random
import string
from datetime import date
import os
from pathlib import Path, PurePath

import platform, socket, re, uuid, json, psutil, logging

def worker(x):
    return x * x

def parallel(n_cores, input_args):
    with mp.Pool(n_cores) as pool:
        res = pool.map(worker, input_args)
    return res


class InputOutput(object):
    """
    Class object for saving/loading files, collecting metadata of batches, etc.

    Examples of use:
        io = InputOutput(directory='results', verbose=True)
        io = InputOutput(directory=r'C://path/to/results', verbose=True)

        io.init_save_dir(sub_path='test', unique_id=False)
        io.save_graph(graph, 'subdir1/test_graph.pkl')

        io.save_machine_metadata(sub_path='test')

        io.init_load_dir(sub_path='test')
        graph_load = io.load_graph('subdir1/test_graph.pkl')

    """
    def __init__(self, directory='results', verbose=True):
        """

        :param directory: string, relative or absolute. if relative, will set parent saving directory to asope/results
        :param verbose: boolean flag to print status of not
        """
        self.verbose = verbose
        if os.path.isabs(directory):
            path = Path(directory)
        else:
            path = Path(os.getcwd()).parent.joinpath(directory)

        self.path = path
        self.save_path = self.path
        self.load_path = self.path
        return

    def save_graph(self, graph, filename):
        filepath = self.save_path.joinpath(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose: print(f'Saving graph to {filepath}')

        with open(filepath, 'wb') as file:
            dill.dump(graph, file)
        return

    def load_graph(self, filename):
        filepath = self.load_path.joinpath(filename)
        if self.verbose: print(f'Loading graph from {filepath}')
        with open(filepath, 'rb') as file:
            graph = dill.load(file)
        return graph

    @staticmethod
    def unique_id():
        return r"{}_{}".format(date.today().strftime("%Y%m%d"),''.join(random.choice(string.hexdigits) for _ in range(4)))

    def init_save_dir(self, sub_path='data', unique_id=True):
        if sub_path is not None:
            if unique_id:
                curr_dir = r"{}_{}".format(self.unique_id(), sub_path)
            else:
                curr_dir = sub_path
            self.save_path = self.path.joinpath(curr_dir)
        else:
            self.save_path = self.path

        self.save_path.mkdir(parents=True, exist_ok=True)
        return

    def init_load_dir(self, sub_path):
        if sub_path is not None:
            self.load_path = self.path.joinpath(sub_path)
        else:
            self.load_path = self.path

        if not os.path.exists(self.load_path):
            raise FileExistsError("The directory to load from does not exist")
        return

    def save_machine_metadata(self, sub_path=None):
        if sub_path is not None:
            metadata_path = self.path.joinpath(sub_path)
        else:
            metadata_path = self.path
        metadata = self.get_machine_metadata()
        metadata_path.mkdir(parents=True, exist_ok=True)

        filepath = metadata_path.joinpath('metadata.json')
        if self.verbose: print(f'Saving machine metadata to {filepath}')
        with open(filepath, 'w') as file:
            json.dump(metadata, file)
        return

    @staticmethod
    def get_machine_metadata():
        try:
            metadata = dict()
            metadata['platform'] = platform.system()
            metadata['platform-release'] = platform.release()
            metadata['platform-version'] = platform.version()
            metadata['architecture'] = platform.machine()
            metadata['hostname'] = socket.gethostname()
            metadata['ip-address'] = socket.gethostbyname(socket.gethostname())
            metadata['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
            metadata['processor'] = platform.processor()
            metadata['ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
            metadata['cpu-count'] = mp.cpu_count()
            return metadata
        except Exception as e:
            logging.exception(e)

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