"""
Classes for working with options on underlying stocks.

"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from utils.baseclass import BaseDataclass
from utils.enums import OptionRight, OptionType
from utils.misc import is_pos


__all__ = ['BaseBinomialTree', 'BinomialTreeOption', 'BinomialCCROption',
           'BinomialLROption', 'BinomialResult', 'BinomialLRWithGreeks',
           'TrinomialTreeOption', 'BinomialCCRLattice', 'TrinomialLattice']


@dataclass(init=False)
class BaseBinomialTree(ABC, BaseDataclass):
    """
    Base class for computing Binomial trees.

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
    qu : Risk-neutral probability for the up state.
    qd : Risk-neutral probability for the down state.
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
        self.STs = []

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

    @property
    @abstractmethod
    def d(self) -> float:
        """Expected value in the down state."""

    @property
    def qu(self) -> float:
        """Risk-neutral probability for the up state."""
        num = (1 / self.df) - self.d
        den = self.u - self.d
        return num / den

    @property
    def qd(self) -> float:
        """Risk-neutral probability for the down state."""
        return 1 - self.qu

    @property
    def dt(self) -> float:
        """Single time step, in years."""
        return self.T / float(self.N)

    @property
    def df(self) -> float:
        """The discount factor."""
        return math.exp(-self.net_r * self.dt)

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        # Intialise a 2D tree at T=0
        self.STs = [np.array([self.S])]

        # Simulate the possible stock prices path
        for _ in range(self.N):
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
        return payoffs

    def begin_tree_traversal(self) -> float:
        """Calculate payoffs at end node, and discount to present time."""
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self) -> float:
        """Entry point of the pricing implementation."""
        self.init_stock_price_tree()
        dis_payoff = self.begin_tree_traversal()

        # Option value converges to first node
        option_value = dis_payoff[0]
        return option_value


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


class BinomialCCROption(BaseBinomialTree):
    """
    Price an option on a stock using the Binomial CCR tree model.

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


class BinomialLROption(BaseBinomialTree):
    """
    Price an option on a stock using the Binomial CCR tree model.

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
    odd_N : j(n) value = {n, if n is even || n + 1, if n is odd}.
    d1 : (log(S / K) + (r + (σ^2 / 2)) * T) / (σ * √T).
    d2 : (log(S / K) + (r - (σ^2 / 2)) * T) / (σ * √T).
    p : f(d2, j(n))
    pbar : f(d1, j(n))

    Methods
    ----------
    pp_2_inversion(z : float, n : float)
        Peizer and Pratt inversion function.
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
    def odd_N(self) -> float:
        """j(n) value = {n, if n is even || n + 1, if n is odd}."""
        return self.N if self.N % 2 == 0 else self.N + 1

    @property
    def d1(self) -> float:
        """d1 = (log(S / K) + (r + (σ^2 / 2)) * T) / (σ * √T)."""
        S = self.S
        K = self.K
        r = self.net_r
        v = self.vol
        T = self.T
        return (math.log(S / K) + (r + (v**2 / 2)) * T) / (v * math.sqrt(T))

    @property
    def d2(self) -> float:
        """d1 = (log(S / K) + (r - (σ^2 / 2)) * T) / (σ * √T)."""
        S = self.S
        K = self.K
        r = self.net_r
        v = self.vol
        T = self.T
        return (math.log(S / K) + (r - (v**2 / 2)) * T) / (v * math.sqrt(T))

    @property
    def p(self) -> float:
        """p = f(d2, j(n))."""
        return self.pp_2_inversion(self.d2, self.odd_N)

    @property
    def pbar(self) -> float:
        """p' = f(d1, j(n))."""
        return self.pp_2_inversion(self.d1, self.odd_N)

    @property
    def u(self) -> float:
        """Expected value in the up state."""
        return 1 / self.df * self.pbar / self.p

    @property
    def d(self) -> float:
        """Expected value in the down state."""
        return (1 / self.df - (self.p * self.u)) / (1 - self.p)

    @property
    def qu(self) -> float:
        """Risk-neutral probability for the up state."""
        return self.p

    @property
    def qd(self) -> float:
        """Risk-neutral probability for the down state."""
        return 1 - self.p

    @staticmethod
    def pp_2_inversion(z: float, n: float) -> float:
        """Peizer and Pratt inversion function."""
        return 0.5 + math.copysign(1, z) * math.sqrt(
            0.25 - 0.25 * math.exp(
                -((z / (n + 1/3 + 0.1 / (n + 1)))**2) * (n + 1/6)
            )
        )


@dataclass
class BinomialResult:
    """Struct of the price and greeks as computed using a binomial tree."""
    price: float
    delta: float
    gamma: float


class BinomialLRWithGreeks(BinomialLROption):
    """
    Price an option on a stock using the Binomial CCR tree model with Greeks.

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
    odd_N : j(n) value = {n, if n is even || n + 1, if n is odd}.
    d1 : (log(S / K) + (r + (σ^2 / 2)) * T) / (σ * √T).
    d2 : (log(S / K) + (r - (σ^2 / 2)) * T) / (σ * √T).
    p : f(d2, j(n))
    pbar : f(d1, j(n))

    Methods
    ----------
    pp_2_inversion(z : float, n : float)
        Peizer and Pratt inversion function.
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

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        self.STs = [np.array([self.S * self.u / self.d,
                              self.S,
                              self.S * self.d / self.u])]

        # Simulate the possible stock prices path
        for _ in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches * self.u, [prev_branches[-1] * self.d])
            )
            self.STs.append(st) # Add nodes at each time step

    def price(self) -> BinomialResult:
        """Entry point of the pricing implementation."""
        self.init_stock_price_tree()
        dis_payoff = self.begin_tree_traversal()

        # Option value is not in the middle node at t=0
        option_value = dis_payoff[len(dis_payoff) // 2]

        payoff_up = dis_payoff[0]
        payoff_down = dis_payoff[-1]
        S_up = self.STs[0][0]
        S_down = self.STs[0][-1]
        dS_up = S_up - self.S
        dS_down = self.S - S_down

        # Calculate delta value
        dS = S_up - S_down
        dV = payoff_up - payoff_down
        delta = dV / dS

        # Calculate gamma value
        gamma = ((payoff_up - option_value) / dS_up -
                 (option_value - payoff_down) / dS_down) / \
            ((self.S + S_up) / 2 - (self.S + S_down) / 2)

        return BinomialResult(option_value, delta, gamma)


class TrinomialTreeOption(BaseBinomialTree):
    """
    Price an option on a stock using a trinomial tree.

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
    m : Expected value in the middle state.
    qu : Risk-neutral probabiity for the up state.
    qd : Risk-neutral probabiity for the down state.
    qm : Risk-neutral probabiity for the middle state.

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
        return math.exp(self.vol * math.sqrt(2 * self.dt))

    @property
    def d(self) -> float:
        """Expected value in the down state."""
        return 1 / self.u

    @property
    def m(self) -> float:
        """Expected value in the middle state."""
        return 1.

    @property
    def qu(self) -> float:
        """Risk-neutral probability for the up state."""
        S = self.S
        K = self.K
        r = self.net_r
        v = self.vol
        dt = self.dt
        num = math.exp(r * dt/2) - math.exp(-v * math.sqrt(dt/2))
        den = math.exp(v * math.sqrt(dt/2)) - math.exp(-v * math.sqrt(dt/2))
        return math.pow(num / den, 2)

    @property
    def qd(self) -> float:
        """Risk-neutral probability for the down state."""
        S = self.S
        K = self.K
        r = self.net_r
        v = self.vol
        dt = self.dt
        num = math.exp(v * math.sqrt(dt/2)) - math.exp(r * dt/2)
        den = math.exp(v * math.sqrt(dt/2)) - math.exp(-v * math.sqrt(dt/2))
        return math.pow(num / den, 2)

    @property
    def qm(self) -> float:
        """Risk-neutral probability for the middle state."""
        return 1 - self.qu - self.qd

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        self.STs = [np.array([self.S])]

        for _ in range(self.N):
            prev_nodes = self.STs[-1]
            st = np.concatenate(
                (prev_nodes * self.u, [prev_nodes[-1] * self.m,
                                       prev_nodes[-1] * self.d]))
            self.STs.append(st)

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
            payoffs = (payoffs[: -2] * self.qu +
                       payoffs[1: -1] * self.qm +
                       payoffs[2:] * self.qd) * self.df

            # Payoffs from exercising early if American type option
            if self.option_type == OptionType.American:
                payoffs = self.check_early_exercise(payoffs, n)
        return payoffs


class BinomialCCRLattice(BinomialCCROption):
    """
    Price an option on a stock using a Binomial CCR model lattice.

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
    M : Size of data structure.

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
    def M(self) -> float:
        """Size of data structure."""
        return 2 * self.N + 1

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S * self.u**self.N

        for i in range(1, self.M):
            self.STs[i] = self.STs[i - 1] * self.d

    def init_payoffs_tree(self) -> List[float]:
        """Returns the payoffs when the option expires at maturity."""
        odd_nodes = self.STs[::2] # Take odd nodes only
        if self.option_right is OptionRight.Call:
            return np.maximum(0, odd_nodes - self.K)
        return np.maximum(0, self.K - odd_nodes)

    def check_early_exercise(self, payoffs: List[float],
                             node: int) -> List[float]:
        """
        Returns the maximum payoff values between exercising early and not
        exercising the option at all.
        """
        self.STs = self.STs[1: -1] # Shortens ends of the list
        odd_STs = self.STs[::2] # Take odd nodes only
        if self.option_right is OptionRight.Call:
            return np.maximum(payoffs, odd_STs - self.K)
        return np.maximum(payoffs, self.K - odd_STs)


class TrinomialLattice(TrinomialTreeOption):
    """
    Price an option on a stock using a trinomial tree model lattice.

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
    m : Expected value in the middle state.
    qu : Risk-neutral probabiity for the up state.
    qd : Risk-neutral probabiity for the down state.
    qm : Risk-neutral probabiity for the middle state.
    M : Size of data structure.

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
    def M(self) -> float:
        """Size of data structure."""
        return 2 * self.N + 1

    def init_stock_price_tree(self) -> None:
        """Initialise stock prices for each node."""
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S * self.u**self.N

        for i in range(1, self.M):
            self.STs[i] = self.STs[i - 1] * self.d

    def init_payoffs_tree(self) -> List[float]:
        """Returns the payoffs when the option expires at maturity."""
        if self.option_right is OptionRight.Call:
            return np.maximum(0, self.STs - self.K)
        return np.maximum(0, self.K - self.STs)

    def check_early_exercise(self, payoffs: List[float],
                             node: int) -> List[float]:
        """
        Returns the maximum payoff values between exercising early and not
        exercising the option at all.
        """
        self.STs = self.STs[1: -1] # Shortens ends of the list
        if self.option_right is OptionRight.Call:
            return np.maximum(payoffs, self.STs - self.K)
        return np.maximum(payoffs, self.K - self.STs)
