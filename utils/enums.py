"""
Enumeration utility classes.
"""

from enum import Enum, auto

__all__ = ['OptionRight', 'OptionType']


class PPEnum(Enum):
    """Enum with prettier printing."""

    def __repr__(self) -> str:
        return super().__repr__().split('.')[1].split(':')[0]

    def __str__(self) -> str:
        return super().__str__().split('.')[1]


class OptionRight(PPEnum):
    """Right of an option."""
    Call = auto()
    Put = auto()


class OptionType(PPEnum):
    """Type of an option."""
    European = auto()
    American = auto()
