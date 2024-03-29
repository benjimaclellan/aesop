#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
from config import config as configuration


def register_node_types_all(cls):
    """Register all children node-types into a global config dictionary for use elsewhere.
    Data structure is a dict, with keys as the parent nodetypes and values as a nested dict, mapping class name to
    class
    """
    register_node_types_including_terminals(cls)
    if cls.__base__.__name__ not in configuration.NODE_TYPES_ALL:
        configuration.NODE_TYPES_ALL[cls.__base__.__name__] = dict()
    configuration.NODE_TYPES_ALL[cls.__base__.__name__][cls.__name__] = cls
    return cls


def register_node_types_including_terminals(cls):
    """Register all children node-types into a global config dictionary for use elsewhere.
    Data structure is a dict, with keys as the parent nodetypes and values as a nested dict, mapping class name to
    class
    """
    if cls.__base__.__name__ not in configuration.NODE_TYPES_ALL_WITH_TERMINALS:
        configuration.NODE_TYPES_ALL_WITH_TERMINALS[cls.__base__.__name__] = dict()
    configuration.NODE_TYPES_ALL_WITH_TERMINALS[cls.__base__.__name__][cls.__name__] = cls
    return cls