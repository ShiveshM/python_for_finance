"""
Miscellaneous utility methods.
"""

import logging

import numpy as np


__all__ = ['is_num', 'is_pos']


def is_num(val: (int, float, np.integer, np.float32, np.float64)) -> bool:
    """
    Check if the input value is a non-infinite number.

    Parameters
    ----------
    val : Value to check.

    Returns
    ----------
    is_num : Whether it is a non-infinite number.

    Examples
    ----------
    >>> from utils.misc import is_num
    >>> print(is_num(10))
    True
    >>> print(is_num(None))
    False

    """
    if not isinstance(val, (int, float, np.integer, np.float32, np.float64)):
        logging.debug(f'Found invalid value val={val}')
        return False
    if np.isnan(val) or np.isinf(val):
        logging.debug(f'Found invalid value val={val}')
        return False
    return True


def is_pos(val: (int, float, np.integer, np.float32, np.float64)) -> bool:
    """
    Check if the input value is a non-infinite positive number.

    Parameters
    ----------
    val : Value to check.

    Returns
    ----------
    is_pos : Whether it is a non-infinite positive number.

    Examples
    ----------
    >>> from utils.misc import is_pos
    >>> print(is_pos(10))
    True
    >>> print(is_pos(-10))
    False

    """
    if not is_num(val):
        return False
    if not val >= 0:
        logging.debug(f'Found value >=0 val={val}')
        return False
    return True
