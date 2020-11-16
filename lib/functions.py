import autograd.numpy as np
import dill
import random
import string
from datetime import date
import os
from pathlib import Path, PurePath
import sys

import platform, socket, re, uuid, json, psutil, logging

#%%
import sys

class Tee(object):
    """
    Changes stream of all print statements to both the terminal and a log file (
    """
    def __init__(self, filename, mode):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.close()

class TeeWrapper(object):
    def __init__(self, filename, mode):
        self._ipy = self._check_ipython()
        if self._ipy:
            import IPython.utils
            self.log = IPython.utils.io.Tee(filename, mode=mode, channel='stdout')
        else:
            self.log = Tee(filename, mode)
            sys.stdout = self.log

    def _check_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    def write(self, message):
        self.log.write(message)

    def close(self):
        if self._ipy:
            self.log.close()
        else:
            sys.stdout = self.log.terminal
            self.log.flush()
        return

class InputOutput(object):
    """
    Class object for saving/loading files, collecting metadata of batches, etc.

    Examples of use:
        io = InputOutput(directory='results', verbose=True)
        io = InputOutput(directory=r'C://path/to/results', verbose=True)

        io.init_save_dir(sub_path='test', unique_id=False)
        io.save_object(graph, 'subdir1/test_graph.pkl')

        io.save_machine_metadata(sub_path='test')

        io.init_load_dir(sub_path='test')
        graph_load = io.load_object('subdir1/test_graph.pkl')

    """
    def __init__(self, directory=None, verbose=True):
        """

        :param directory: string, relative or absolute. if relative, will set parent saving directory to folder called 'asope_results' at the same level as asope
        :param verbose: boolean flag to print status of not
        """
        default_path = 'asope_data'

        self.verbose = verbose
        if directory is not None:
            assert type(directory) is str
            if os.path.isabs(directory):
                path = Path(directory)
            else:
                path = Path(os.getcwd()).parent.parent.joinpath(default_path, directory)
        else:
            path = Path(os.getcwd()).parent.parent.joinpath(default_path)

        self.path = path
        self.save_path = self.path
        self.load_path = self.path
        return

    def init_logging(self):
        self._tee = TeeWrapper(self.join_to_save_path('stdout_logging.log'), 'w')
        return

    def close_logging(self):
        self._tee.close()

    def save_object(self, object_to_save, filename):
        filepath = self.join_to_save_path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose: print(f'Saving graph to {filepath}')

        with open(filepath, 'wb') as file:
            dill.dump(object_to_save, file)
        return

    def load_object(self, filename):
        filepath = self.join_to_load_path(filename)
        if self.verbose: print(f'Loading graph from {filepath}')
        with open(filepath, 'rb') as file:
            object_to_load = dill.load(file)
        return object_to_load

    @staticmethod
    def unique_id():
        return r"{}_{}".format(date.today().strftime("%Y%m%d"),''.join(random.choice(string.hexdigits) for _ in range(4)))

    def init_save_dir(self, sub_path=None, unique_id=True):
        if sub_path is not None:
            if unique_id:
                curr_dir = r"{}_{}".format(self.unique_id(), sub_path)
            else:
                curr_dir = sub_path
            self.save_path = self.path.joinpath(curr_dir)
        else:
            if unique_id:
                self.save_path = self.path.joinpath(self.unique_id())
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
            if os.path.isabs(sub_path):
                metadata_path = sub_path
            else:
                metadata_path = self.join_to_save_path(sub_path)
        else:
            metadata_path = self.path
        metadata = self.get_machine_metadata()
        metadata_path.mkdir(parents=True, exist_ok=True)

        filepath = metadata_path.joinpath('metadata.json')
        if self.verbose: print(f'Saving machine metadata to {filepath}')
        with open(filepath, 'w') as file:
            json.dump(metadata, file, indent=2)
        return

    def load_json(self, filename):
        filepath = self.join_to_load_path(filename)
        if self.verbose: print(f'Loading JSON from {filepath}')
        with open(filepath, 'rb') as file:
            dictionary = json.load(file)
        return dictionary

    def save_json(self, dictionary, filename):
        filepath = self.join_to_save_path(filename)
        if self.verbose: print(f'Saving JSON to {filepath}')
        with open(filepath, 'w') as file:
            json.dump(dictionary, file, indent=2)
        return

    def save_fig(self, fig, filename):
        filepath = self.join_to_save_path(filename)
        if self.verbose: print(f'Saving figure to {filepath}')
        kwargs = {'dpi': 150, 'transparent': False}
        fig.savefig(filepath, bbox_inches='tight', **kwargs)
        return

    def join_to_save_path(self, filename):
        return self.save_path.joinpath(filename)

    def join_to_load_path(self, filename):
        return self.load_path.joinpath(filename)

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
            metadata['cpu-count'] = psutil.cpu_count()
            return metadata
        except Exception as e:
            print("Exception getting machine metadata")

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