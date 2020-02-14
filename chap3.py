#! /usr/bin/env python3

"""Utility functions from Chapter 3 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 3 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.

"""

from typing import Callable, Tuple


__all__ = ['incremental', 'bisection', 'newtons', 'secant', 'scipy',
           'scipy_general']

STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def incremental() -> None:
    """Incremental search root-finding algorithm."""
    import numpy as np

    def incremental_search(func: Callable[[float], float],
                           bounds: Tuple[int],
                           increment: float) -> (float, int):
        """Incremental search algorithm.

        Parameters
        ----------
        func : Callable[[float], float]
            Function to evaluate
        bounds : Tuple[int]
            Lower and upper bounds
        increment : float
            Incremental value in searching

        Returns
        ----------
        root : float
            Value at the root
        n_iter : int
            Number of interations used

        """
        low, high = bounds

        # Start from lower boundary
        x = low
        y = x + increment
        f_x = func(low)
        f_y = func(y)

        # Increment until a sign change occurs
        n_iter = 0
        while np.sign(f_x) == np.sign(f_y):
            if x >= y:
                return x - increment, n_iter

            x = y
            f_x = f_y
            y = x + increment
            f_y = func(y)
            n_iter += 1

        if f_x == 0:
            return x, n_iter
        elif f_y == 0:
            return y, n_iter
        else:
            return (x + y)/2, n_iter

    # Find the root of a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    root, iterations = incremental_search(y, (-5, 5), 0.001)
    print(STR_FMT.format('root', root))
    print(STR_FMT.format('iterations', iterations))


def bisection() -> None:
    """Bisection method root-finding algorithm.

    Notes
    ----------
    Suppose we know two points of an interval a and b, and that f(a) <0 and
    f(b) > 0 lie along a continuos function. Taking the midpoint of this
    interval as c, where c = (a + b) / 2; the bisection method then evaluates
    this value as f(c). In the next evaluation, c is replaced as either a or b
    accordingly. With the new interval shortened, the bisection method repeats
    with the same evaluation to determine the next value of c.

    The biggest advantage of this method is that it is guaranteed to converge
    to an approximation of the root. Also to note is that this method does not
    require knowledge of the derivative of the unknown function. It's major
    drawback is that it is more computationally intensive than algorithms which
    utilise the derivative information. It also requires a good approximation
    of the search boundary.

    """
    def bisection_method(func: Callable[[float], float],
                         bounds: Tuple[int],
                         tol: float = 1E-5,
                         max_iter: int = 1E6) -> (float, int):
        """Bisection method algorithm.

        Parameters
        ----------
        func : Callable[[float], float]
            Function to evaluate
        bounds : Tuple[int]
            Lower and upper bounds
        tol : float, optional
            Tolerance to compute to (the default is 1E-5)
        max_iter : int, optional
            Maximum number of iterations (the default is 1E6)

        Returns
        ----------
        root : float
            Value at the root
        n_iter : int
            Number of interations used

        """
        a, b = bounds

        # Iterate until stopping condition
        n_iter = 1
        while n_iter <= max_iter:
            # Calculate next value to evaluate
            c = (a + b) / 2

            # Check if root is found to tolerance
            if func(c) == 0 or abs(a - b) / 2 < tol:
                return c, n_iter

            # Adjust search boundary
            if func(c) < 0:
                a = c
            else:
                b = c
            n_iter += 1
        return c, n_iter

    # Find the root of a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    root, iterations = bisection_method(y, (-5, 5))
    print(STR_FMT.format('root', root))
    print(STR_FMT.format('iterations', iterations))


def newtons() -> None:
    """Newton-Raphson method root-finding algorithm.

    Notes
    ----------
    This method uses an iterative procedure to solve for the root using
    information about the derivative of a function. The approximation to the
    next value of x is given as:

                        x' = x - f(x)/f'(x)

    Here, the tangent line intersects the x axis at x', which produces y = 0.
    This also represents a first-order Taylor expansion about x', such that the
    new point solves f(x' + Δx) = 0.

    An initial guess value is required to compute the values of f(x) and f'(x).
    The rate of convergence is quadratic, which is considered to be extremely
    fast. A drawback is that it does not gaurantee global convergence to the
    solution - e.g. when the solution contains more than one root or if the
    algorithm arrives at a local extremum. It is required that the input
    function be differentiable.

    """
    def newtons_method(func: Callable[[float], float],
                       df: Callable[[float], float],
                       seed: float,
                       tol: float = 1E-5,
                       max_iter: int = 1E6) -> (float, int):
        """Newton's method algorithm.

        Parameters
        ----------
        func : Callable[[float], float]
            Function to evaluate
        df : Callable[[float], float]
            Derivative of the function to evaluate
        seed : float
            Initial guess
        tol : float, optional
            Tolerance to compute to (the default is 1E-5)
        max_iter : int, optional
            Maximum number of iterations (the default is 1E6)

        Returns
        ----------
        root : float
            Value at the root
        n_iter : int
            Number of interations used

        """
        x = seed

        # Iterate until stopping condition
        n_iter = 1
        while n_iter <= max_iter:
            # Calculate next value to evaluate
            x1 = x - (func(x) / df(x))

            # Check if root is found to tolerance
            if abs(x1 - x) < tol:
                return x1, n_iter

            # Adjust search boundary
            x = x1
            n_iter += 1
        return None, n_iter

    # Find the root of a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    dy = lambda x: (3 * x**2) + 4 * x
    seed = 5
    root, iterations = newtons_method(y, dy, seed)
    print(STR_FMT.format('root', root))
    print(STR_FMT.format('iterations', iterations))


def secant() -> None:
    """Secant method root-finding algorithm.

    Notes
    ----------
    This method uses secant lines to find the root, which is a straight line
    that intersects two points of a curve. By successively drawing such secant
    lines, the root of the function can be approximated.

    An initial guess of the two x axis values, a and b, is required. A secant
    line, y, is drawn from f(b) to f(a) and intersects at the point c on the x
    axis such that:

                y = (c - b) * (f(b) - f(a)) / (b - a) + f(b)

    Therefore,

                   c = b - f(b) * (b - a) / (f(b) - f(a))

    On the next iteration, a and b will take on the values b and c,
    respectively. This method then repeats itself, terminating when the maximum
    number of iterations is reached, or the difference between b and c has
    reached a specified tolerance level.

    The rate of convergence is considered to be superlinear. It converges much
    faster than the bisection method and slower than Newton's method.

    """
    def secant_method(func: Callable[[float], float],
                      bounds: Tuple[int],
                      tol: float = 1E-5,
                      max_iter: int = 1E6) -> (float, int):
        """Secant method algorithm.

        Parameters
        ----------
        func : Callable[[float], float]
            Function to evaluate
        bounds : Tuple[int]
            Lower and upper bounds
        tol : float, optional
            Tolerance to compute to (the default is 1E-5)
        max_iter : int, optional
            Maximum number of iterations (the default is 1E6)

        Returns
        ----------
        root : float
            Value at the root
        n_iter : int
            Number of interations used

        """
        a, b = bounds

        # Iterate until stopping condition
        n_iter = 1
        while n_iter <= max_iter:
            # Calculate next value to evaluate
            c = b - func(b) * ((b - a) / (func(b) - func(a)))

            # Check if root is found to tolerance
            if abs(c - b) < tol:
                return c, n_iter

            # Adjust search boundary
            a = b
            b = c
            n_iter += 1
        return None, n_iter

    # Find the root of a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    root, iterations = secant_method(y, (-5, 5))
    print(STR_FMT.format('root', root))
    print(STR_FMT.format('iterations', iterations))


def scipy() -> None:
    """Scipy root-finding scalar functions [1]_.

    References
    ----------
    .. [1] SciPy, "Optimization and Root Finding (scipy.optimize) - SciPy
           Reference Guide",
           https://docs.scipy.org/doc/scipy/reference/optimize.html

    """
    import scipy.optimize as optimize

    # Define a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    dy = lambda x: (3 * x**2) + 4 * x

    # Call method: bisect(f, a, b[, args, xtol, rtol, maxiter, ...])
    print(STR_FMT.format('Bisection Method:',
                         optimize.bisect(y, -5, 5, xtol=1E-5)))

    # Call method: newton(func, x0[, fprime, args, tol, ...])
    print(STR_FMT.format('Newton\'s Method:',
                         optimize.newton(y, 5, fprime=dy)))

    # When fprime=None, the secant method is used
    print(STR_FMT.format('Secant Method:',
                         optimize.newton(y, 5)))

    # Call method: brentq(f, a, b[, args, xtol, rtol, maxiter, ...])
    # This method combines the bisection root-finding method, secant method,
    # and inverse quadratic interpolation
    print(STR_FMT.format('Brent\'s Method:',
                         optimize.brentq(y, -5, 5)))


def scipy_general() -> None:
    """Scipy general multidimensional non-linear solvers [1]_.

    References
    ----------
    .. [1] SciPy, "Optimization and Root Finding (scipy.optimize) - SciPy
           Reference Guide",
           https://docs.scipy.org/doc/scipy/reference/optimize.html

    """
    import scipy.optimize as optimize

    # Define a test function
    y = lambda x: x**3 + 2 * x**2 - 5
    dy = lambda x: (3 * x**2) + 4 * x

    # Call method: fsolve(func, x0[, args, fprime, ...])
    # Find the roots of a function
    print(STR_FMT.format('optimize.fsolve',
                         optimize.fsolve(y, 5, fprime=dy)))

    # Call method: root(fun, x0[, args, method, jac, tol, ...])
    # Find the root of a vector function
    print(STR_FMT.format('optimize.root',
                         optimize.root(y, 5)))


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Mastering Python for Finance - Chapter 3'
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
