import importlib

def import_config(config_filename):
    ConfigObject = getattr(importlib.import_module(config_filename), "Config")
    return ConfigObject()