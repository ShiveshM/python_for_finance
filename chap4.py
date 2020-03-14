#! /usr/bin/env python3

"""
Utility functions from Chapter 4 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 4 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.

"""

import math


__all__ = ['european_option', 'american_option', 'cox_ross_rubinstein',
           'leisen_reimer', 'greeks', 'trinomial_tree', 'binomial_lattice']

IMGDIR = './img/chap4/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def european_option() -> None:
    r"""
    Pricing European options using a binomial tree.

    Notes
    ----------
    Consider a two-step binomial tree. A non-dividend paying stock starts at
    $50, and, in each of the two time steps, the stock may go up by 20% or go
    down by 20%. Suppose the risk-free rate is 5% per annum and that the time
    to maturity, T, is 2 years. Find the value of a European put option with a
    strike K of $52.

    Using a binomial tree, the nodes will have stock price values:

                            S_uu = $72, p_uu = $0
                          /
                S_u = $60
              /           \
    S_0 = $50               S_ud = S_du = $48, p_ud = p_du = $52 - $48 = $4
              \           /
                S_d = $40
                          \
                            S_dd = $32, p_dd = $52 - $32 = $20

    with final payoff values p_uu, p_ud, and p_dd. We then traverse the
    binomial tree backward to the current time, and after discounting the
    risk-free rate, we will obtain the present value of the option.

    In the case of investing in stocks by risk-neutral probability, the payoff
    from holding the stock and taking into account the up and down state
    possibilities, would be equal to the continuously compounded risk-free rate
    expected in the next time step, as follows

                          e^{rt} = q * u + (1 - q)d

    The risk-neutral probability q of investing in the stock is then

                          q = (e^{rt} - d) / (u - d)

    Note that for forward contracts, in the risk-neutral sense, the expected
    growth rate from holding a forward contract is zero, and so the
    risk-neutral probability can be written as

                             q = (1 - d) / (u - d)

    The present value of the put option can be priced as:

          p_t = e^{-r(T - t)}[ 0(q)^2 + 2(4)(q)(1 - q) + 20(1 - q)^2 ]

    """
    from utils.option import BinomialTreeOption

    S = 50
    K = 52
    T = 2
    option_right = 'Put'
    option_type = 'European'
    r = 0.05
    N = 2
    u = 1.2
    d = 0.8
    t = T / N

    # Calculate the risk-neutral probability
    q = (math.exp(r * t) - d) / (u - d)
    print(STR_FMT.format('risk-free probability, q', '{:.2f}'.format(q)))

    # Calculate the value of the option at each node
    p0 = math.exp(-r * T) * ((2 * 4 * q * (1 - q)) + (20 * (1 - q)**2))
    pu = math.exp(-r * t) * (4 * (1 - q))
    pd = math.exp(-r * t) * ((4 * q) + (20 * (1 - q)))
    print(STR_FMT.format('p0', '${:.2f}'.format(p0)))
    print(STR_FMT.format('pu', '${:.2f}'.format(pu)))
    print(STR_FMT.format('pd', '${:.2f}'.format(pd)))

    # Do the same using the BinomialTreeOption class
    eu_option = BinomialTreeOption(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        N=N, pu=(u - 1), pd=(1 - d),
    )
    print(STR_FMT.format('European option put price at T0:',
                         '${:.2f}'.format(eu_option.price())))
    print(STR_FMT.format('eu_option', f'{eu_option}'))


def american_option() -> None:
    """
    Pricing American options using a binomial tree.

    Notes
    ----------
    Unlike European options, which can only be exercised at maturity, American
    options can be exercised at any time during their lifetime.

    Since American options can be exercised at any time, this added flexibility
    compared to European options increases their value in certain
    circumstances. For an American call option on an underlying asset that does
    not pay dividends, there might not be an extra value over its European call
    option counterpart.

    Because of the time value of money, it costs more to exercise the American
    call option today before the expiration at the strike price than at a
    future time with the same strike price. For an in-the-money American call
    option, exercising the option early loses the benefit of protection against
    adverse price movement below the strike price, as well as its intrinsic
    time value. With no entitlement of dividend payments, there are no
    incentives to exercise American call options early.

    """
    from utils.option import BinomialTreeOption

    S = 50
    K = 52
    T = 2
    option_right = 'Put'
    option_type = 'European'
    r = 0.05
    N = 2
    u = 1.2
    d = 0.8

    # Use the BinomialTreeOption class
    am_option = BinomialTreeOption(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        N=N, pu=(u - 1), pd=(1 - d),
    )

    # Since the American option has extra flexibility, it is priced higher than
    # European options in certain circumstances
    print(STR_FMT.format('American option put price at T0:',
                         '${:.2f}'.format(am_option.price())))
    print(STR_FMT.format('am_option', f'{am_option}'))


def cox_ross_rubinstein() -> None:
    """
    Pricing options using a binomial tree with underlying stock modelled using
    the Cox-Ross-Rubinstein model.

    Notes
    ----------
    In the above example, we assumed that the underlying stock price would
    increase by 20% and decrease by 20% in the rejective `u` up state and `d`
    down state. The Cox-Ross-Rubinstein (CCR) model proposes that, over a short
    period of time in the risk-neutral world, the binomial model matches the
    mean and variance of the underlying stock. The volatility of the underlying
    stock is taken into account as follows

                                u = e^{ σ √{Δt} }
                        d = 1 / u = e^{ -σ √{Δt} }

    """
    from utils.option import BinomialCCROption

    S = 50
    K = 52
    option_right = 'Put'
    option_type = 'European'
    T = 2
    r = 0.05
    vol = 0.3
    N = 2

    # European option
    option = BinomialCCROption(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        vol=vol, N=N
    )
    print(STR_FMT.format('option', f'{option}'))
    print(STR_FMT.format('European option put price at T0:',
                         '${:.2f}'.format(option.price())))

    # American option
    option.option_type = 'American'
    print(STR_FMT.format('American option put price at T0:',
                         '${:.2f}'.format(option.price())))


def leisen_reimer() -> None:
    """
    The Leisen-Reimer (LR) tree model [1].

    Notes
    ----------
    LR tree model is a binomial tree model with the purpose of approximating
    the Black-Scholes solution as the number of steps increases. The nodes do
    not recombine at every alternative step and it uses an inversion formula to
    achieve better accuracy during tree traversal.

    Here is used method two of the Peizer and Pratt inversion function f with
    the following characteristic parameters:

        f(z, j(n)) = 0.5 ∓ √[0.25 - 0.25 *
            exp{-(z / (n + (1/3) + (0.1 / (n + 1))))^2 * (n + (1/6))}]

                j(n) = {n, if n is even || n + 1, if n is odd}

                p' = f(d1, j(n))           p = f(d2, j(n))

             d1 = (log(S / K) + (r + (σ^2 / 2)) * T) / (σ * √T)
             d2 = (log(S / K) + (r - (σ^2 / 2)) * T) / (σ * √T)

                        u = exp{r * Δt} * (p' / p)
                   d = (exp{r * Δy} - (p * u)) / (1 - p)

    [1] Leisen, D. & Reimer, M. "Binomial Models for Option Valuation -
        Examining and Improving Convergence"
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5976

    """
    from utils.option import BinomialLROption

    S = 50
    K = 52
    option_right = 'Put'
    option_type = 'European'
    T = 2
    r = 0.05
    vol = 0.3
    N = 4

    # European option
    option = BinomialLROption(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        vol=vol, N=N
    )
    print(STR_FMT.format('option', f'{option}'))
    print(STR_FMT.format('European option put price at T0:',
                         '${:.2f}'.format(option.price())))

    # American option
    option.option_type = 'American'
    print(STR_FMT.format('American option put price at T0:',
                         '${:.2f}'.format(option.price())))


def greeks() -> None:
    r"""
    The Greeks for free.

    Notes
    ----------
    Here we add an additional layer of nodes around our original two-step tree
    to make it a four-step tree, which extends backward in time. Even with the
    additional terminal payoff nodes, all node values will contain the same
    information as our original two-step tree. Out option value of interest is
    now located in the middle of the tree at t=0:

    t=-2     t=-1    t=0   t=1   t=2
                                S_2uu
                               /
                          S_1u
                         /     \
                    S_0u/d      S_uu
                   /     \     /
             S_-1u         S_u
         u /       \     /     \
    S_-2              S0        S_ud
         d \       /     \     /
             S_-1d         S_d
                   \     /     \
                    S_0d/u      S_dd
                         \     /
                          S_1d
                               \
                                S_2dd

    Notice that at t=0 there exists two additional nodes' worth of information
    that we can use to compute the delta formula, as follows

                Δ = (v_{up} - v_{down}) / (S_0ud - S_0du)

    The delta formula states that the difference in the option prices in the up
    and down state is represented as a unit of the difference between the
    respective stock prices at time t=0.

    Conversely, gamma can be computed as follows

    γ = (((v_{up} - v_0) / (S_0ud - S0)) / ((v_0 - v_{down}) / (S0 - S_0du))) /
                   (((S0 + S_0ud) / 2) - ((S0 + S_0du) / 2))

    The gamma formula states that the difference of deltas between the option
    prices in the up node and the down node against the initial node value are
    computed as a unit of the differences in price of the stock at the
    respective states.

    """
    from utils.option import BinomialLRWithGreeks

    S = 50
    K = 52
    option_right = 'Put'
    option_type = 'European'
    T = 2
    r = 0.05
    vol = 0.3
    N = 300

    # European option
    option = BinomialLRWithGreeks(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        vol=vol, N=N
    )
    print(STR_FMT.format('option', f'{option}'))
    print(STR_FMT.format('European option put price at T0:',
                         f'{option.price()}'))

    # European option
    option.option_right = 'Call'
    print(STR_FMT.format('European option call price at T0:',
                         f'{option.price()}'))

    # American option
    option.option_type = 'American'
    option.option_right = 'Put'
    print(STR_FMT.format('American option put price at T0:',
                         f'{option.price()}'))

    option.option_right = 'Call'
    print(STR_FMT.format('American option call price at T0:',
                         f'{option.price()}'))


def trinomial_tree() -> None:
    r"""
    Trinomial trees in option pricing.

    Notes
    ----------
    In a trinomial tree, each node leads to three nodes in the next step.
    Besides having an up and down state, the middle node of the trinomial tree
    indicates no change in state.

    Let's consider a Boyle trinomial tree, where the tree is calibrated so that
    the probability of up, down, and flat movements, u, d, and m with
    risk-neutral probabilities q_u, q_d, and q_m are as follows

        u = e^{ σ √{Δt} }       m = ud = 1      d = 1 / u = e^{ -σ √{Δt} }

            q_u = ((exp{r * Δt / 2} - exp{-σ * √{Δt / 2}}) /
                   (exp{σ * √{Δt / 2}} - exp{-σ * √{Δt / 2}}))**2

            q_d = ((exp{σ * √{Δt / 2}} - exp{r * Δt / 2}) /
                   (exp{σ * √{Δt / 2}} - exp{-σ * √{Δt / 2}}))**2

                           q_m = 1 - q_u - q_d

    In general, with an increased number of nodes to process, a trinomial tree
    gives better accuracy than the binomial tree when fewer time steps are
    modelled, saving on computation speed and resources.

    """
    from utils.option import TrinomialTreeOption

    S = 50
    K = 52
    option_right = 'Put'
    option_type = 'European'
    T = 2
    r = 0.05
    vol = 0.3
    N = 2

    # European option
    option = TrinomialTreeOption(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        vol=vol, N=N
    )
    print(STR_FMT.format('option', f'{option}'))
    print(STR_FMT.format('European option put price at T0:',
                         '${:.2f}'.format(option.price())))

    # American option
    option.option_type = 'American'
    print(STR_FMT.format('American option put price at T0:',
                         '${:.2f}'.format(option.price())))


def binomial_lattice() -> None:
    """
    Lattices in option pricing.

    Notes
    ----------
    In binomial trees, each nodes recombines to every alternative node. In
    trinomial trees, each node recombines at every other node. This property of
    recombining trees can also be represented as lattices to save memory
    without recomputing and storing recombined nodes.

    In a binomial CCR tree, at every alternate up and down nodes, the prices
    recombine to the same probability of ud = 1. Now the tree can be
    represented as a single list; [Suu, Su, S, Sd, Sdd].

    For an N-step binomial tree, a list of size 2N + 1 is required to contain
    the information on the underlying stock prices. For European option
    pricing, the odd nodes of payoffs from the list represent the option value
    upon maturity. The tree traverses backwards to obtain the option value. For
    American option pricing, as the tree traverses backward, both ends of the
    list shrink, and the odd nodes represent the associated stock prices for
    any step. Payoffs from earlier exercise can then be taken into account.

    """
    from utils.option import BinomialCCRLattice

    S = 50
    K = 52
    option_right = 'Put'
    option_type = 'European'
    T = 2
    r = 0.05
    vol = 0.3
    N = 2

    # European option
    option = BinomialCCRLattice(
        S, K, option_right=option_right, option_type=option_type, T=T, r=r,
        vol=vol, N=N
    )
    print(STR_FMT.format('option', f'{option}'))
    print(STR_FMT.format('European option put price at T0:',
                         '${:.2f}'.format(option.price())))

    # American option
    option.option_type = 'American'
    print(STR_FMT.format('American option put price at T0:',
                         '${:.2f}'.format(option.price())))


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Mastering Python for Finance - Chapter 4'
    )
    parser.add_argument('functions', nargs='*', help=f'Choose from {__all__}')
    args = parser.parse_args()

    functions = args.functions if args.functions else __all__
    for f in functions:
        if f not in __all__:
            raise ValueError(f'Invalid function "{f}" (choose from {__all__})')
        print('------', f'\nRunning "{f}"')
        globals()[f]()
        print('------')


if __name__ == "__main__":
    main()
