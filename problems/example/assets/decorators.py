#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
import config.config as configuration

def register_node_types_all(cls):
    """Register all children node-types into a global config dictionary for use elsewhere.
    Data structure is a dict, with keys as the parent nodetypes and values as a nested dict, mapping class name to
    class
    """
    if cls.__base__.__name__ not in configuration.NODE_TYPES_ALL:
        configuration.NODE_TYPES_ALL[cls.__base__.__name__] = dict()
    configuration.NODE_TYPES_ALL[cls.__base__.__name__][cls.__name__] = cls
    return cls

