"""
Utility classes for working with options on underlying stocks.

"""

import math
from dataclasses import dataclass

import numpy as np

from utils.baseclass import BaseDataclass
from utils.enums import OptionRight, OptionType
from utils.misc import is_pos


__all__ = ['StockOption', 'BinomialEuropeanOption']


@dataclass
class StockOption(BaseDataclass):
    """
    Object for storing parameters of a stock option.

    Attributes
    ----------
    S : Stock price today (or at the time of evaluation).
    K : Strike price.
    T : Time to maturity.
    r : Risk-free interest rate.
    vol : Volatility.
    div : Dividend yield.
    option_right : Right of the option.
    N : Number of time steps.
    pu : Probability in up state.
    pd : Probability in down state.
    dt : Single time step, in years.
    df : The discount factor.
    STs : The stock prices tree.

    """
    N: int = 2
    pu: float = 0.
    pd: float = 0.

    def __post_init__(self):
        """Post initialisation processing."""
        # Declare the stock prices tree
        self.STs = []

    @property
    def dt(self) -> float:
        """Single time step, in years."""
        return self.T / float(self.N)

    @property
    def df(self) -> float:
        """The discount factor."""
        return math.exp(-(self.r - self.div) * self.dt)


class BinomialEuropeanOption(StockOption):
    """
    Price a European option on a stock using the binomial tree model.

    Attributes
    ----------
    S : Stock price today (or at the time of evaluation).
    K : Strike price.
    T : Time to maturity.
    option_right : Right of the option.
    r : Risk-free interest rate.
    vol : Volatility.
    div : Dividend yield.
    N : Number of time steps.
    pu : Probability in up state.
    pd : Probability in down state.
    dt : Single time step, in years.
    df : The discount factor.
    STs : The stock prices tree.
    M : Number of terminal nodes of a tree.
    u : Expected value in the up state.
    d : Expected value in the down state.
    qu : Risk-neutral probabiity for the up state.
    qd: Risk-neutral probabiity for the down state.

    Methods
    ----------
    init_stock_price_tree()
        Initialise stock prices for each node.
    init_payoffs_tree()
        Returns the payoffs when the option expires at terminal nodes.
    traverse_tree(payoffs : ndarray)
        Calculate discounted payoffs.
    begin_tree_traversal()
        Calculate payoffs at end node, and discount to present time.
    price()
        Entry point of the pricing implementation.

    """

    @property
    def option_type(self) -> OptionType:
        """Return the type of the option."""
        return OptionType.European

    @property
    def M(self) -> int:
        """Number of terminal nodes of a tree."""
        return self.N + 1

    @property
    def u(self) -> float:
        """Expected value in the up state."""
        return 1 + self.pu

    @property
    def d(self) -> float:
        """Expected value in the down state."""
        return 1 - self.pd

    @property
    def qu(self) -> float:
        """Risk-neutral probabiity for the up state."""
        num = math.exp((self.r - self.div) * self.dt) - self.d
        den = self.u - self.d
        return num / den

    @property
    def qd(self) -> float:
        """Risk-neutral probabiity for the down state."""
        return 1 - self.qu

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        # Intialise terminal price nodes to zeros
        self.STs = np.zeros(self.M)

        # Calculate expected stock prices for each end node
        for i in range(self.M):
            self.STs[i] = self.S * (self.u**(self.N - i)) * (self.d**i)

    def init_payoffs_tree(self) -> 'ndarray':
        """Returns the payoffs when the option expires at terminal nodes."""
        if self.option_right is OptionRight.Call:
            return np.maximum(0, self.STs - self.K)
        return np.maximum(0, self.K - self.STs)

    def traverse_tree(self, payoffs: 'ndarray') -> float:
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
        for _ in range(self.N):
            payoffs = (payoffs[:-1] * self.qu +
                       payoffs[1:] * self.qd) * self.df

        # Option value converges to first node
        dis_payoff = payoffs[0]
        return dis_payoff

    def begin_tree_traversal(self) -> float:
        """Calculate payoffs at end node, and discount to present time."""
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self) -> float:
        """Entry point of the pricing implementation."""
        self.init_stock_price_tree()
        dis_payoff = self.begin_tree_traversal()
        return dis_payoff
