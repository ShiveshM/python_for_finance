"""
Base dataclass for pricing classes.
"""

from dataclasses import dataclass

from utils.enums import OptionRight
from utils.misc import is_num, is_pos


__all__ = ['BaseDataclass']


@dataclass
class BaseDataclass:
    S: float
    K: float
    option_right: (str, OptionRight)
    T: float = 1
    r: float = 0.05
    vol: float = 0.
    div: float = 0.

    @property
    def option_right(self) -> OptionRight:
        """Right of the option; either 'call' or 'put'."""
        return self._option_right

    @option_right.setter
    def option_right(self, val: (str, OptionRight)) -> None:
        """Set the option_right of the option; either 'call' or 'put'."""
        if isinstance(val, str):
            if not hasattr(OptionRight, val):
                or_names = [x.name for x in OptionRight]
                raise ValueError(f'Invalid str {val}, expected {or_names}')
            self._option_right = OptionRight[val]
        elif isinstance(val, OptionRight):
            self._option_right = val
        else:
            raise TypeError(
                f'Expected str or OptionRight, instead got type {type(val)}!'
            )
