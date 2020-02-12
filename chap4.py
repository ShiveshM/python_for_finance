#! /usr/bin/env python3

"""Utility functions from Chapter 4 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 4 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.
"""

import math


__all__ = ['european_option']

IMGDIR = './img/chap4/'
STR_FMT = '{0}\n{1}\n'


def european_option() -> None:
    """Pricing European options using a binomial tree.

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
                S_u = $60
                          \
                            S_dd = $32, p_dd = $52 - $32 = $20

    with final payoff values p_uu, p_ud, and p_dd. We then traverse the
    binomial tree backward to the current time, and after discounting the
    risk-free rate, we will obtain the present value of the option.

    In the case of investing in stocks by risk-neutral probability, the payoff
    from holding the stock and taking into account the up and down state
    possibilities, would be equal to the continuously compounded risk-free rate
    expected in the next time step, as follows
                        e^{rt} = qu + (1 - q)d
    The risk-neutral probability q of investing in the stock is then
                      q = (e^{rt} - d) / (u - d)

    Note that for forward contracts, in the risk-neutral sense, the expected
    growth rate from holding a forward contract is zero, and so the
    risk-neutral probability can be written as
                         q = (1 - d) / (u - d)

    The present value of the put option can be priced as:
          p_t = e^{-r(T - t)}[ 0(q)^2 + 2(48)(q)(1 - q) + 20(1 - q)^2 ]

    """
    r = 0.05
    T = 2
    t = T / 2
    u = 1.2
    d = 0.8

    q = (math.exp(r * t) - d) / (u - d)
    print(STR_FMT.format('risk-free probability, q', '{:.2f}'.format(q)))

    p0 = math.exp(-r * T) * ((2 * 4 * q * (1 - q)) + (20 * (1 - q)**2))
    pu = math.exp(-r * t) * (4 * (1 - q))
    pd = math.exp(-r * t) * ((4 * q) + (20 * (1 - q)))
    print(STR_FMT.format('p0', '${:.2f}'.format(p0)))
    print(STR_FMT.format('pu', '${:.2f}'.format(pu)))
    print(STR_FMT.format('pd', '${:.2f}'.format(pd)))


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
