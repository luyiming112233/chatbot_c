# -*- coding: utf-8 -*-
"""
CONFIG
------
configuration
"""
from configparser import ConfigParser

__config = None


def get_config(config_file_path='theme_distinguish/conf/config.txt'):
    """
    singleton object generator
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config
