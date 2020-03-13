"""
Classes for working with options on underlying stocks.

"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from utils.baseclass import BaseBinomialTree
from utils.enums import OptionRight, OptionType
from utils.misc import is_pos


__all__ = ['BinomialTreeOption', 'BinomialCCROption']


@dataclass(init=False)
class BinomialTreeOption(BaseBinomialTree):
    """
    Price an option using assumed risk-neutral probabilities

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
    pu: float = 0.5
    pd: float = 0.5

    def __init__(self, S: float, K: float, option_right: (str, OptionRight),
                 option_type: (str, OptionType), T: float = 1.,
                 r: float = 0.05, vol: float = 0., div: float = 0.,
                 N: int = 2, pu: float = 0.5, pd: float = 0.5):
        super().__init__(S, K, option_right, option_type, T, r, vol, div, N)
        self.pu = pu
        self.pd = pd

    @property
    def pu(self) -> float:
        """Risk free probability of up move."""
        return self._pu

    @pu.setter
    def pu(self, val: float) -> None:
        """Set the risk free probability of up move."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        self._pu = float(val)

    @property
    def pd(self) -> float:
        """Risk free probability of down move."""
        return self._pd

    @pd.setter
    def pd(self, val: float) -> None:
        """Set the risk free probability of down move."""
        if not is_pos(val):
            raise ValueError(
                f'Expected non-negative float, instead got {val} with type '
                f'{type(val)}!'
            )
        self._pd = float(val)

    @property
    def u(self) -> float:
        """Expected value in the up state."""
        return 1 + self.pu

    @property
    def d(self) -> float:
        """Expected value in the down state."""
        return 1 - self.pd

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        # Intialise a 2D tree at T=0
        self.STs = [np.array([self.S])]

        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches * self.u, [prev_branches[-1] * self.d])
            )
            self.STs.append(st) # Add nodes at each time step

    def init_payoffs_tree(self) -> List[float]:
        """Returns the payoffs when the option expires at maturity."""
        if self.option_right is OptionRight.Call:
            return np.maximum(0, self.STs[self.N] - self.K)
        return np.maximum(0, self.K - self.STs[self.N])

    def check_early_exercise(self, payoffs: List[float],
                             node: int) -> List[float]:
        """
        Returns the maximum payoff values between exercising early and not
        exercising the option at all.
        """
        if self.option_right is OptionRight.Call:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])

    def traverse_tree(self, payoffs: List[float]) -> List[float]:
        """
        Calculate discounted payoffs.

        Starting from the time the option expires, traverse backwards and
        calculate discounted payoffs at each node. Includes invocation which
        checks if it is optimal to exercise early at every step.

        Parameters
        ----------
        payoffs : List of payoffs at the end node.

        Returns
        ----------
        dis_payoff : Discounted payoff.

        """
        for n in reversed(range(self.N)):
            # Payoffs from NOT exercising the option
            payoffs = (payoffs[:-1] * self.qu +
                       payoffs[1:] * self.qd) * self.df

            # Payoffs from exercising early if American type option
            if self.option_type == OptionType.American:
                payoffs = self.check_early_exercise(payoffs, n)

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


class BinomialCCROption(BaseBinomialTree):
    """
    Price an option on a stock using the binomial CCR tree model.

    Attributes
    ----------
    S : Stock price today (or at the time of evaluation).
    K : Strike price.
    T : Time to maturity.
    option_right : Right of the option.
    option_type : Type of option
    r : Risk-free interest rate.
    vol : Volatility.
    div : Dividend yield.
    N : Number of time steps.
    pu : Probability in up state.
    pd : Probability in down state.
    dt : Single time step, in years.
    df : The discount factor.
    u : Expected value in the up state.
    d : Expected value in the down state.
    qu : Risk-neutral probabiity for the up state.
    qd : Risk-neutral probabiity for the down state.
    STs : The stock prices tree.

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

    @property
    def u(self) -> float:
        """Expected value in the up state."""
        return math.exp(self.vol * math.sqrt(self.dt))

    @property
    def d(self) -> float:
        """Expected value in the down state."""
        return 1 / self.u

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        # Intialise a 2D tree at T=0
        self.STs = [np.array([self.S])]

        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches * self.u, [prev_branches[-1] * self.d])
            )
            self.STs.append(st) # Add nodes at each time step

    def init_payoffs_tree(self) -> List[float]:
        """Returns the payoffs when the option expires at maturity."""
        if self.option_right is OptionRight.Call:
            return np.maximum(0, self.STs[self.N] - self.K)
        return np.maximum(0, self.K - self.STs[self.N])

    def check_early_exercise(self, payoffs: List[float],
                             node: int) -> List[float]:
        """
        Returns the maximum payoff values between exercising early and not
        exercising the option at all.
        """
        if self.option_right is OptionRight.Call:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])

    def traverse_tree(self, payoffs: List[float]) -> List[float]:
        """
        Calculate discounted payoffs.

        Starting from the time the option expires, traverse backwards and
        calculate discounted payoffs at each node. Includes invocation which
        checks if it is optimal to exercise early at every step.

        Parameters
        ----------
        payoffs : List of payoffs at the end node.

        Returns
        ----------
        dis_payoff : Discounted payoff.

        """
        for n in reversed(range(self.N)):
            # Payoffs from NOT exercising the option
            payoffs = (payoffs[:-1] * self.qu +
                       payoffs[1:] * self.qd) * self.df

            # Payoffs from exercising early if American type option
            if self.option_type == OptionType.American:
                payoffs = self.check_early_exercise(payoffs, n)

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
