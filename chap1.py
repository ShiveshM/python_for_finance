#! /usr/bin/env python3

"""Utility functions from Chapter 1 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 1 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.

Functions:

    version() -> None

"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quandl


__all__ = ['version', 'plot_time_series']

IMGDIR = './img/chap1/'
STR_FMT = '{0}\n{1}\n'


def version() -> None:
    """Print version information."""
    print(f'Python {sys.version}')
    print(f'Numpy {np.__version__}')
    print(f'Pandas {pd.__version__}')
    print(f'Quandl {quandl.version.VERSION}')


def plot_time_series() -> None:
    """Plot a time series using data from Quandl."""
    quandl.read_key()

    # Get the data
    df = quandl.get('EURONEXT/ABN')
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))
    df.plot()
    plt.savefig(IMGDIR+'dataset.png', bbox_inches='tight')

    # Extract the daily closing price and volume
    prices = df['Last']
    volumes = df['Volume']
    print(STR_FMT.format('prices.head()', prices.head()))
    print(STR_FMT.format('volumes.tail()', volumes.tail()))
    print(STR_FMT.format('type(volumes)', type(volumes)))

    # Plot the prices and volumes
    # Top plot consisting of daily closing price
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    top.plot(prices.index, prices, label='Last')
    plt.title('ABN Last Price from {low} - {high}'.format(
            low=prices.index[0].year, high=prices.index[-1].year))
    plt.legend(loc=2)

    # The bottom plot consisting of daily trading volume
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    bottom.bar(volumes.index, volumes)
    plt.title('ABN Daily Trading Volume')

    # Save figure
    plt.gcf().set_size_inches(12, 8)
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(IMGDIR+'time_series.png', bbox_inches='tight')


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Mastering Python for Finance - Chapter 1'
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
