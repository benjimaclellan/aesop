#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import config.config as configuration


def register_evolution_operators(func):
    """Register all evolution operators into a global config dictionary for use elsewhere
    """
    configuration.EVOLUTION_OPERATORS[func.__name__] = func
    return func



def register_node_types(cls):
    """Register a node-type into a global config dictionary for use elsewhere
    """
    configuration.NODE_TYPES[cls.__name__] = cls
    return cls