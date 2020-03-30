import importlib
import os
import sys
from importlib import import_module
import os.path

# %% splits a path string into parts (successive folders)
def splitpath(path, maxdepth=20):
    (head, tail) = os.path.split(path)
    return splitpath(head, maxdepth - 1) + [tail] \
        if maxdepth and head and head != path \
        else [head or tail]

# %% imports a config (saved as a Python class from the path)
def import_config(package_path):
    path, name = os.path.split(package_path)
    sub, _ = os.path.splitext(name)
    toplevel = splitpath(path)[-1]

    sys.path.append(path)

    module_object = import_module('{:s}.{:s}'.format(toplevel,sub))

    target_class = getattr(module_object, "Config")
    return target_class()