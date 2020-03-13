"""
Base dataclass for pricing classes.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from utils.enums import OptionRight, OptionType
from utils.misc import is_num, is_pos

__all__ = ['BaseDataclass', 'BaseBinomialTree']


@dataclass(init=False)
class BaseDataclass:
    """
    Base dataclass for storing relavant parameters.

    Attributes
    ----------
    S : Stock price today (or at the time of evaluation).
    K : Strike price.
    option_right : Right of the option.
    option_type : Type of the option.
    T : Time to maturity, in years.
    r : Risk-free interest rate.
    vol : Volatility.
    div : Dividend yield.
    net_r : Net risk free rate.

    """
    S: float
    K: float
    option_right: (str, OptionRight)
    option_type: (str, OptionType)
    T: float = 1
    r: float = 0.05
    vol: float = 0.
    div: float = 0.

    def __init__(self, S: float, K: float, option_right: (str, OptionRight),
                 option_type: (str, OptionType), T: float = 1.,
                 r: float = 0.05, vol: float = 0., div: float = 0.):
        self.S = S
        self.K = K
        self.option_right = option_right
        self.option_type = option_type
        self.T = T
        self.r = r
        self.vol = vol
        self.div = div

    @property
    def S(self) -> float:
        """
        Spot price of underlying (in the case of a future it represents the
        value of the future underlying).
        """
        return self._S

    @S.setter
    def S(self, val: float) -> None:
        """Set the spot price of the underlying."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        self._S = float(val)

    @property
    def K(self) -> float:
        """Strike price."""
        return self._K

    @K.setter
    def K(self, val: float) -> None:
        """Set the strike price."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        self._K = float(val)

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

    @property
    def option_type(self) -> OptionType:
        """Right of the option; either 'call' or 'put'."""
        return self._option_type

    @option_type.setter
    def option_type(self, val: (str, OptionType)) -> None:
        """Set the option_type of the option; either 'call' or 'put'."""
        if isinstance(val, str):
            if not hasattr(OptionType, val):
                or_names = [x.name for x in OptionType]
                raise ValueError(f'Invalid str {val}, expected {or_names}')
            self._option_type = OptionType[val]
        elif isinstance(val, OptionType):
            self._option_type = val
        else:
            raise TypeError(
                f'Expected str or OptionType, instead got type {type(val)}!'
            )

    @property
    def T(self) -> float:
        """Time to maturity."""
        return self._T

    @T.setter
    def T(self, val: float) -> None:
        """Set the time to maturity."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        if val == 0:
            raise ValueError(f'T cannot be set to zero, got {val}')
        self._T = float(val)

    @property
    def r(self) -> float:
        """Risk-free rate."""
        return self._r

    @r.setter
    def r(self, val: float) -> None:
        """Set the risk-free rate."""
        if not is_num(val):
            raise ValueError(
                f'Expected float, instead got {val} with type {type(val)}!'
            )
        self._r = float(val)

    @property
    def vol(self) -> float:
        """Risk-free rate."""
        return self._vol

    @vol.setter
    def vol(self, val: float) -> None:
        """Set the risk-free rate."""
        if not is_num(val):
            raise ValueError(
                f'Expected float, instead got {val} with type {type(val)}!'
            )
        self._vol = float(val)

    @property
    def div(self) -> float:
        """Time to maturity."""
        return self._div

    @div.setter
    def div(self, val: float) -> None:
        """Set the time to maturity."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        self._div = float(val)

    @property
    def net_r(self) -> float:
        """Net risk free rate."""
        return self.r - self.div


@dataclass(init=False)
class BaseBinomialTree(ABC, BaseDataclass):
    """
    Base class for computing binomial trees.

    Attributes
    ----------
    S : Stock price today (or at the time of evaluation).
    K : Strike price.
    option_right : Right of the option.
    option_type : Type of the option.
    T : Time to maturity, in years.
    r : Risk-free interest rate.
    vol : Volatility.
    div : Dividend yield.
    net_r : Net risk free rate.
    N : Number of time steps.
    u : Expected value in the up state.
    d : Expected value in the down state.
    qu : Risk-neutral probabiity for the up state.
    qd : Risk-neutral probabiity for the down state.
    dt : Single time step, in years.
    df : The discount factor.

    Methods
    ----------
    init_stock_price_tree()
        Initialise stock prices for each node.
    init_payoffs_tree()
        Returns the payoffs when the option expires at terminal nodes.
    traverse_tree(payoffs : List[float])
        Calculate discounted payoffs.
    begin_tree_traversal()
        Calculate payoffs at end node, and discount to present time.
    price()
        Entry point of the pricing implementation.

    """
    N: int = 2

    def __init__(self, S: float, K: float, option_right: (str, OptionRight),
                 option_type: (str, OptionType), T: float = 1.,
                 r: float = 0.05, vol: float = 0., div: float = 0.,
                 N: int = 2):
        super().__init__(S, K, option_right, option_type, T, r, vol, div)
        self.N = N

    @property
    def N(self) -> int:
        """Number of increments."""
        return self._N

    @N.setter
    def N(self, val: int) -> None:
        """Set the number of increments."""
        if not isinstance(val, (int, np.int)):
            raise ValueError(
                f'Expected integer, instead got {val} with type {type(val)}'
            )
        if not is_pos(val):
            raise ValueError(f'Expected non-negative int, instead got {val}')
        self._N = val

    @property
    @abstractmethod
    def u(self) -> float:
        """Expected value in the up state."""
        pass

    @property
    @abstractmethod
    def d(self) -> float:
        """Expected value in the down state."""
        pass

    @property
    def qu(self) -> float:
        """Risk-neutral probabiity for the up state."""
        num = (1 / self.df) - self.d
        den = self.u - self.d
        return num / den

    @property
    def qd(self) -> float:
        """Risk-neutral probabiity for the down state."""
        return 1 - self.qu

    @property
    def dt(self) -> float:
        """Single time step, in years."""
        return self.T / float(self.N)

    @property
    def df(self) -> float:
        """The discount factor."""
        return math.exp(-self.net_r * self.dt)

    @abstractmethod
    def init_stock_price_tree(self) -> None:
        """Calculate stock prices at each node."""
        pass

    @abstractmethod
    def init_payoffs_tree(self) -> List[float]:
        """Returns the payoffs at maturity."""
        pass

    @abstractmethod
    def traverse_tree(self, payoffs: List[float]) -> float:
        """
        Calculate discounted payoffs.

        Starting from the time the option expires, traverse backwards and
        calculate discounted payoffs at each node.

        Parameters
        ----------
        payoffs : List of payoffs at the end node.

        Returns
        ----------
        dis_payoff : Discounted payoff.

        """
        pass

    @abstractmethod
    def begin_tree_traversal(self) -> float:
        """Calculate payoffs at end node, and discount to present time."""
        pass

    @abstractmethod
    def price(self) -> float:
        """Calculate the premium of the option."""
        pass
