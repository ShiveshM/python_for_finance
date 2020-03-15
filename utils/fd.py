"""
Classes for calculating option prices using finite difference methods.

"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from utils.baseclass import BaseDataclass
from utils.enums import OptionRight, OptionType
from utils.misc import is_pos


__all__ = ['BaseFiniteDifferences', 'FDExplicitEu']


@dataclass(init=False)
class BaseFiniteDifferences(ABC, BaseDataclass):
    """
    Base class for computing finite differences.

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
    M : Number of underlying steps.
    Smax : Maximum possible price of underlying.
    dS : Finite change in S per unit time.
    dt : Finite change in t per unit stock.

    Methods
    ----------
    setup_grid()
        Setup the grid.
    setup_boundary_conditions()
        Setup the boundary conditions.
    setup_coefficients()
        Setup the coefficients.
    traverse_grid()
        Iterate the grid backwards in time.
    interpolate()
        Use piecewise linear interpolation on the initial grid column to get
        the closest price at S0.
    price()
        Entry point of the pricing implementation.

    """
    N: int = 2
    M: int = 2
    Smax: float = 1.

    def __init__(self, S: float, K: float, option_right: (str, OptionRight),
                 option_type: (str, OptionType), T: float = 1.,
                 r: float = 0.05, vol: float = 0., div: float = 0.,
                 N: int = 2, M: int = 2, Smax: float = 1.):
        super().__init__(S, K, option_right, option_type, T, r, vol, div)
        self.N = N
        self.M = M
        self.Smax = Smax

    @property
    def N(self) -> int:
        """Number of time increments."""
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
        self.t_steps = np.arange(val)

    @property
    def M(self) -> int:
        """Number of underlying increments."""
        return self._M

    @M.setter
    def M(self, val: int) -> None:
        """Set the number of underlying increments."""
        if not isinstance(val, (int, np.int)):
            raise ValueError(
                f'Expected integer, instead got {val} with type {type(val)}'
            )
        if not is_pos(val):
            raise ValueError(f'Expected non-negative int, instead got {val}')
        self._M = val
        self.s_steps = np.arange(val)

    @property
    def Smax(self) -> int:
        """Number of underlying increments."""
        return self._Smax

    @Smax.setter
    def Smax(self, val: int) -> None:
        """Set the number of underlying increments."""
        if not isinstance(val, (int, np.int)):
            raise ValueError(
                f'Expected integer, instead got {val} with type {type(val)}'
            )
        if not is_pos(val):
            raise ValueError(f'Expected non-negative int, instead got {val}')
        self._Smax = val

    @property
    def dS(self) -> float:
        """Finite change in S per unit time."""
        return self.Smax / self.M

    @property
    def dt(self) -> float:
        """Finite change in t per unit stock."""
        return self.T / self.N

    @abstractmethod
    def setup_boundary_conditions(self) -> None:
        """Setup the boundary conditions"""

    @abstractmethod
    def setup_coefficients(self) -> None:
        """Setup the coefficients."""

    @abstractmethod
    def traverse_grid(self) -> None:
        """Iterate the grid backwards in time."""

    def setup_grid(self) -> None:
        """Setup the grid."""
        self.grid = np.zeros(shape=(self.M + 1, self.N + 1))
        self.boundary_conds = np.linspace(0, self.Smax, self.M + 1)

    def interpolate(self) -> float:
        """
        Use piecewise linear interpolation on the initial grid column to get
        the closest price at S0.

        """
        return np.interp(self.S, self.boundary_conds, self.grid[:, 0])

    def price(self) -> float:
        """Entry point of the pricing implementation."""
        self.setup_grid()
        self.setup_boundary_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


class FDExplicitEu(BaseFiniteDifferences):
    """
    Base class for computing finite differences.

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
    M : Number of underlying steps.
    Smax : Maximum possible price of underlying.
    dS : Finite change in S per unit time.
    dt : Finite change in t per unit stock.

    Methods
    ----------
    setup_grid()
        Setup the grid.
    setup_boundary_conditions()
        Setup the boundary conditions
    setup_coefficients()
        Setup the coefficients.
    traverse_grid()
        Iterate the grid backwards in time.
    interpolate()
        Use piecewise linear interpolation on the initial grid column to get
        the closest price at S0.
    price()
        Entry point of the pricing implementation.

    """

    def setup_boundary_conditions(self) -> None:
        """Setup the boundary conditions"""
        if self.option_right == OptionRight.Call:
            self.grid[:, -1] = np.maximum(0, self.boundary_conds - self.K)
            self.grid[-1,: -1] = (self.Smax - self.K) * \
                np.exp(-self.net_r * self.dt * (self.N - self.t_steps))
        else:
            self.grid[:, -1] = np.maximum(0, self.K - self.boundary_conds)
            self.grid[0,: -1] = (self.K - self.boundary_conds[0]) * \
                np.exp(-self.net_r * self.dt * (self.N - self.t_steps))

    def setup_coefficients(self) -> None:
        """Setup the coefficients."""
        self.a = (1/2) * self.dt * (self.vol**2 * self.s_steps**2 -
                                    self.net_r * self.s_steps)
        self.b = 1 - self.dt * (self.vol**2 * self.s_steps**2 + self.net_r)
        self.c = (1/2) * self.dt * (self.vol**2 * self.s_steps**2 +
                                    self.net_r * self.s_steps)

    def traverse_grid(self) -> None:
        """Iterate the grid backwards in time."""
        for i_t in reversed(self.t_steps):
            for i_s in range(1, self.M):
                self.grid[i_s, i_t] = \
                    self.a[i_s] * self.grid[i_s - 1, i_t + 1] + \
                    self.b[i_s] * self.grid[i_s, i_t + 1] + \
                    self.c[i_s] * self.grid[i_s + 1, i_t + 1]
