#! /usr/bin/env python3

"""Utility functions from Chapter 1 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 1 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.

"""

import pandas as pd
import matplotlib.pyplot as plt
import quandl


__all__ = ['version', 'plot_time_series', 'plot_candlestick',
           'time_series_analytics', 'qq_plot', 'correlation',
           'plot_correlation', 'sma', 'ema']

IMGDIR = './img/chap1/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def version() -> None:
    """Print version information."""
    import sys
    import numpy as np
    print(f'Python {sys.version}')
    print(f'Numpy {np.__version__}')
    print(f'Pandas {pd.__version__}')
    print(f'Quandl {quandl.version.VERSION}')


def plot_time_series() -> None:
    """Plot a time series using data from Quandl."""
    quandl.read_key()

    # Get data of ABN Amro
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


def plot_candlestick() -> None:
    """Plot a candlestick chart using data from Quandl."""
    import matplotlib.dates as mdates
    from mpl_finance import candlestick_ohlc

    quandl.read_key()

    # Get data of ABN Amro
    df = quandl.get(
        'EURONEXT/ABN',
        start_date='2018-07-01',
        end_date='2018-07-31'
    )

    # Convert index to mpl format date and extract open, high, low, and close
    df['Date'] = df.index.map(mdates.date2num)
    df_ohlc = df[['Date', 'Open', 'High', 'Low', 'Last']]
    print(STR_FMT.format('df_ohlc.head()', df_ohlc.head()))
    print(STR_FMT.format('df_ohlc.tail()', df_ohlc.tail()))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(formatter)
    candlestick_ohlc(
        ax,
        df_ohlc.values,
        width=0.8,
        colorup='green',
        colordown='red'
    )

    # Save figure
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(IMGDIR+'candlestick.png', bbox_inches='tight')


def time_series_analytics() -> None:
    """Visualise some statistical proprties of time series."""
    quandl.read_key()

    # Get data of ABN Amro
    df = quandl.get('EURONEXT/ABN', column_index=4)
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))

    # Calculate and plot the percentage daily returns
    daily_changes = df.pct_change(periods=1)
    print(STR_FMT.format('daily_changes.describe()', daily_changes.describe()))
    daily_changes.plot()
    plt.savefig(IMGDIR+'pct_change.png', bbox_inches='tight')

    # Calculate and plot the cumulative returns
    # Equivalent to "df / df['Last'][0] - 1"
    df_cumprod = (daily_changes + 1).cumprod() - 1
    df_cumprod.plot()
    plt.savefig(IMGDIR+'cum_return.png', bbox_inches='tight')

    # Calculate and plot a histogram
    daily_changes.hist(bins=50, figsize=(8, 4))
    plt.savefig(IMGDIR+'hist.png', bbox_inches='tight')

    # Calculate and plot standard deviation / volaility over one month
    df_filled = df.asfreq('D', method='ffill') # Pad missing entries
    df_returns = df_filled.pct_change()
    df_std = df_returns.rolling(window=30, min_periods=30).std()
    df_std.plot()
    plt.savefig(IMGDIR+'volatility.png', bbox_inches='tight')


def qq_plot() -> None:
    """
    Create a Q-Q (quantile-quantile) plot.

    Notes
    ----------
    This is a graphical method for comparing two probability distributions by
    plotting their quantiles against each other, for example to test if the
    daily changes of a stock are distributed normally.

    """
    from scipy import stats

    quandl.read_key()

    # Get the daily changes data
    df = quandl.get('EURONEXT/ABN', column_index=4)
    daily_changes = df.pct_change(periods=1).dropna()
    print(STR_FMT.format('daily_changes.describe()', daily_changes.describe()))

    # Create the Q-Q plot against a normal distribution
    # Note that stats.probplot is the same as a Q-Q plot, however probabilities
    # are shown in the scale of the theoretical distribution (x-axis) and the
    # y-axis contains unscaled quantiles of the sample data.
    fig, ax = plt.subplots(figsize=(8, 4))
    help(stats.probplot)
    stats.probplot(daily_changes['Last'], dist='norm', plot=ax)
    fig.savefig(IMGDIR+'qq_plot.png', bbox_inches='tight')


def correlation() -> None:
    """Downloading multiple time series data and display their correlation."""
    quandl.read_key()

    # Get data for ABN Amro, Banco Santander, and Kas Bank
    df = quandl.get(
        ['EURONEXT/ABN', 'EURONEXT/SANTA', 'EURONEXT/KA'],
        column_index=4, collapse='monthly', start_date='2016-01-01',
        end_date='2017-12-31'
    )
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))
    print(STR_FMT.format('df.describe()', df.describe()))

    # Plot
    df.plot()
    plt.savefig(IMGDIR+'multiple_data.png', bbox_inches='tight')

    # Compute the correlation for the daily changes
    corr = df.pct_change().corr(method='pearson')
    print(STR_FMT.format('corr', corr))


def plot_correlation() -> None:
    """Visualise the correlation between two datasets."""
    quandl.read_key()

    # Get data for ABN Amro, Banco Santander, and Kas Bank
    df = quandl.get(
        ['EURONEXT/ABN', 'EURONEXT/SANTA'],
        column_index=4, start_date='2016-01-01', end_date='2017-12-31'
    )
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))
    print(STR_FMT.format('df.describe()', df.describe()))

    # Compute the daily changes and window size for rolling
    df_filled = df.asfreq('D', method='ffill')
    daily_changes = df_filled.pct_change()
    abn_returns = daily_changes['EURONEXT/ABN - Last']
    santa_returns = daily_changes['EURONEXT/SANTA - Last']
    window = len(df_filled.index) // 2
    print(STR_FMT.format('window', window))
    print(STR_FMT.format('abn_returns.describe()', abn_returns.describe()))
    print(STR_FMT.format('santa_returns.describe()', santa_returns.describe()))

    # Compute the correlation with a rolling window
    df_corrs = abn_returns\
        .rolling(window=window, min_periods=window)\
        .corr(other=santa_returns)\
        .dropna()
    df_corrs.plot(figsize=(12, 8))
    plt.savefig(IMGDIR+'correlation.png', bbox_inches='tight')


def sma() -> None:
    """Simple moving average of a time series."""
    quandl.read_key()

    # Get data of ABN Amro
    df = quandl.get('EURONEXT/ABN', column_index=4)
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))

    # Fill in missing values on a daily basis
    df_filled = df.asfreq('D', method='ffill')
    df_last = df['Last']

    # Calculate the SMA for a 5-day and 30-day window
    series_short = df_last.rolling(window=5, min_periods=5).mean()
    series_long = df_last.rolling(window=30, min_periods=30).mean()

    # Plot the long and short window SMAs
    df_sma = pd.DataFrame(columns=['short', 'long'])
    df_sma['short'] = series_short
    df_sma['long'] = series_long
    df_sma.plot(figsize=(12, 8))
    plt.savefig(IMGDIR+'sma.png', bbox_inches='tight')


def ema() -> None:
    """Exponential moving average of a time series."""
    quandl.read_key()

    # Get data of ABN Amro
    df = quandl.get('EURONEXT/ABN', column_index=4)
    print(STR_FMT.format('df.head()', df.head()))
    print(STR_FMT.format('df.tail()', df.tail()))

    # Fill in missing values on a daily basis
    df_filled = df.asfreq('D', method='ffill')
    df_last = df['Last']

    # Calculate the EMA with a decay spanning 5-days and 30-days
    # The ewm() method provides exponential weighted functions
    help(df_last.ewm)
    series_short = df_last.ewm(span=5).mean()
    series_long = df_last.ewm(span=30).mean()

    # Plot the long and short window EMAs
    df_sma = pd.DataFrame(columns=['short', 'long'])
    df_sma['short'] = series_short
    df_sma['long'] = series_long
    df_sma.plot(figsize=(12, 8))
    plt.savefig(IMGDIR+'ema.png', bbox_inches='tight')


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
