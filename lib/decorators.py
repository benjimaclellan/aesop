#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import config.config as configuration


def register_evolution_operators(cls):
    """
    Register all evolution operators into a global config dictionary for use elsewhere
    """
    configuration.EVOLUTION_OPERATORS[cls.__name__] = cls
    return cls


def register_crossover_operators(cls):
    """
    Register all crossover operators into global config dictionary for use elsewhere
    Technically, this is an evolution operator as well, but crossover operators require a different API
    """
    configuration.CROSSOVER_OPERATORS[cls.__name__] = cls


def register_node_types(cls):
    """
    Register a node-type into a global config dictionary for use elsewhere
    """
    configuration.NODE_TYPES[cls.__name__] = cls
    return cls