#! /usr/bin/env python3

"""
Utility functions from Chapter 4 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 4 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.

"""

import math


__all__ = ['european_option', 'american_option', 'cox_ross_rubinstein',
           'leisen_reimer']

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
    stock is taken into account as follows:

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
    """
    The Greeks for free.

    Notes
    ----------

    """
    pass


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
