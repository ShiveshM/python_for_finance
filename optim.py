#! /usr/bin/env python3

"""
Optimization algorithms.

"""

from typing import Callable, List, Tuple

import numpy as np

# from numdifftools import Jacobian, Hessian # These methods are slower than the implementations below


__all__ = ["vanilla_gradient_descent", "newtons", "levenberg_marquardt"]


def gradient(func: Callable[[float], float], x: [float]) -> [float]:
    """
    Compute the gradient of a scalar function at a point using finite difference techniques.

    Uses a central difference scheme.
                 [Dₜ f]ⁿ = (fⁿ⁺¹⸍² - fⁿ⁻¹⸍²) / Δx    (≈ df(xn)/dx)

    Parameters
    ----------
    func : Function to evaluate.
    x : Value at which to evaluate the gradient.

    Returns
    ----------
    grad : Gradient ∇𝐟(𝘅)

    """
    x = np.asarray(x)
    dx = x.size
    x = x.astype(float)
    f = func(x)

    tol = np.spacing(1) ** (1 / 3)

    h = tol * np.maximum(abs(x), 1)
    x_minus_h = x - h
    x_plus_h = x + h
    deltaX = x_plus_h - x_minus_h
    fx = np.zeros(dx)

    for k in range(dx):
        xx = x.copy()
        xx[k] = x_plus_h[k]
        fplus = func(xx)

        xx[k] = x_minus_h[k]
        fminus = func(xx)

        fx[k] = (fplus - fminus) / deltaX[k]

    return fx


def jacobian(func: Callable[[float], List[float]], x: [float]) -> [[float]]:
    """
    Compute the Jacobian of a function at a point using finite difference techniques.

    Function takes in n arguments and returns m arguments.

    Parameters
    ----------
    func : Function to evaluate.
    x : Value at which to evaluate the Jacobian.

    Returns
    ----------
    Jacobian : [∇^T 𝐟(𝘅)...]

    """
    x = np.asarray(x)
    dx = x.size
    f = func(x)
    df = f.size
    fx = np.zeros((df, dx))

    for k in range(df):
        fx[k] = gradient(lambda z: func(z)[k], x)

    return fx


def hessian(func: Callable[[float], float], x: [float]) -> [[float]]:
    """
    Compute the Hessian of a scalar function at a point using finite difference techniques.

    Parameters
    ----------
    func : Function to evaluate.
    x : Value at which to evaluate the Hessian.

    Returns
    ----------
    Hessian : Jacobian(∇𝐟(𝘅))

    """
    x = np.asarray(x)
    dx = x.size

    grad = lambda z: gradient(func, z)
    fx = jacobian(grad, x)
    return fx


def vanilla_gradient_descent() -> None:
    """Gradient descent minimization algorithm"""

    def vanilla(
        func: Callable[[float], float],
        step_size: float,
        seed: [float],
        tol: float = 1e-7,
        max_iter: float = 1e5,
    ) -> (float, int):
        """
        Vanilla gradient descent algorithm.

        Gradient algorithm where the step size is fixed.

        Parameters
        ----------
        func : Function to evaluate.
        step_size : Relative size of steps to take on each iteration (relative to the gradient).
        seed : Initial guess.
        tol : optional
            Tolerance to compute to.
        max_iter : optional
            Maximum number of iterations.

        Returns
        ----------
        x : Value at the minumum.
        n_iter : Number of iterations used.

        """
        x0 = seed
        y0 = func(x0)
        grad0 = gradient(func, x0)
        # grad0 = Jacobian(func)(x0)[0]

        n_iter = 1
        while np.sum(np.absolute(grad0)) > tol:
            x1 = x0 - step_size * grad0

            if n_iter > max_iter:
                return x1, n_iter

            y1 = func(x1)
            grad1 = gradient(func, x1)
            # grad1 = Jacobian(func)(x1)[0]

            x0 = x1
            y0 = y1
            grad0 = grad1
            n_iter += 1
        return x0, n_iter

    # Find the minimum of a test function
    # Minimum should be [-0.70710677,  0.70710677] - success
    # This method is sensitive to the step size, which must be chosen carefully otherwise this method may not converge.
    y = lambda x: 7 * float(x[0]) * x[1] / np.exp(x[0] ** 2 + x[1] ** 2)
    minx, iterations = vanilla(y, 0.383, [-3.0, 3.0])
    print(f"{minx=}")
    print(f"{iterations=}")

    # Find the minimum of a test function
    # Minimum should be [1.0, 1.0] - failure
    # This method is also extremely slow for functions which have more complex curvature.
    y = lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    minx, iterations = vanilla(y, 0.3, [1.0, 100.0])
    print(f"{minx=}")
    print(f"{iterations=}")


def newtons() -> None:
    """Newton's method to find the minimum of a function."""

    def newtons_method(
        func: Callable[[float], float],
        seed: [float],
        tol: float = 1e-7,
        max_iter: float = 1e5,
    ) -> (float, int):
        """
        Newton's method algorithm to find the minimum of a function.

        Simply uses newton's method to find the root of the derivative of the
        objective function.

        Parameters
        ----------
        func : Function to evaluate.
        seed : Initial guess.
        tol : optional
            Tolerance to compute to.
        max_iter : optional
            Maximum number of iterations.

        Returns
        ----------
        x : Value at the minumum.
        n_iter : Number of interations used

        """
        x0 = seed
        y0 = func(x0)
        grad0 = gradient(func, x0)
        hes0 = hessian(func, x0)
        # grad0 = Jacobian(func)(x0)[0]
        # hes0 = Hessian(func)(x0)
        hinv0 = np.linalg.inv(hes0)

        # Iterate until stopping condition
        n_iter = 1
        while n_iter <= max_iter:
            # Check if root is found to tolerance
            if np.sum(np.absolute(grad0)) < tol:
                return x0, n_iter

            # Calculate next value to evaluate
            x1 = x0 - np.matmul(hinv0, grad0)

            # Adjust search boundary
            x0 = x1
            grad0 = gradient(func, x0)
            hes0 = hessian(func, x0)
            w, v = np.linalg.eig(hes0)
            for ev in w:
                if np.real(ev) < 0:
                    print(
                        f"hessian matrix is not positive semi definite:\n{hes0=}\neigenvalues = {w}"
                    )
            # grad0 = Jacobian(func)(x0)[0]
            # hes0 = Hessian(func)(x0)
            hinv0 = np.linalg.inv(hes0)
            n_iter += 1
        return None, n_iter

    # Find the minimum of a test function
    # Minimum should be [1.0, 1.0] - success
    y = lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    minx, iterations = newtons_method(y, [1.0, 100.0])
    print(f"{minx=}")
    print(f"{iterations=}")

    # Find the minimum of a test function
    # Minimum should be [-0.70710677,  0.70710677] - failure
    # Newton's method fails to find the minimum of the following function.  The
    # hessian matrix in this case is not positive definite, so this method ends
    # up going in the wrong direction.
    y = lambda x: 7 * float(x[0]) * x[1] / np.exp(x[0] ** 2 + x[1] ** 2)
    minx, iterations = newtons_method(y, [-3.0, 3.0])
    print(f"{minx=}")
    print(f"{iterations=}")


def levenberg_marquardt() -> None:
    """Levenberg-Marquardt method to find the minimum of a function."""

    def lm(
        func: Callable[[float], float],
        seed: [float],
        seed_damping: float = 0.1,
        tol: float = 1e-7,
        max_iter: float = 1e5,
    ) -> (float, int):
        """
        Levenberg-Marquardt method to find the minimum of a function.

        This uses a blend of both the gradient descent method and Newton's
        method to achieve stable behaviour across a wider range of functions,
        while mostly being faster than the gradient descent method.  Typically
        this is applied in a specialised case of minimising a least squares
        function, however the implementation here does not assume this.  In a
        least squares problem, approximations can be made to further increase
        performance.

        Parameters
        ----------
        func : Function to evaluate.
        seed : Initial guess.
        seed_damping : Initial damping factor.
        tol : optional
            Tolerance to compute to.
        max_iter : optional
            Maximum number of iterations.

        Returns
        ----------
        x : Value at the minumum.
        n_iter : Number of interations used

        """
        x0 = seed
        y0 = func(x0)
        grad0 = gradient(func, x0)
        hes0 = hessian(func, x0)
        ld0 = seed_damping
        lambda0 = np.identity(len(hes0), dtype=float) * ld0
        # lambda0 = np.diag(np.diag(hes0)) * ld0
        hinv0 = np.linalg.inv(hes0 + lambda0)

        lfactor = 10.0

        # Iterate until stopping condition
        n_iter = 1
        while n_iter <= max_iter:
            # Check if root is found to tolerance
            if np.sum(np.absolute(grad0)) < tol:
                return x0, n_iter

            # Calculate next value to evaluate
            x1 = x0 - hinv0 @ grad0
            y1 = func(x1)
            # print(f"{grad0=}, {hinv0 @ grad0=}, {ld0=}, {x0=}, {y0=}, {x1=}, {y1=}")

            w, v = np.linalg.eig(hinv0)
            for ev in w:
                if np.real(ev) < 0:
                    print(
                        f"{x0=}, {x1=}, {y0=}, {y1=}, hessian matrix is not positive semi definite:\n{hinv0=}\neigenvalues = {w}\n{(x1-x0).T @ (hes0 + lambda0) @ (x1-x0)=}"
                    )

            if y1 > y0:
                ld0 = min(1e2, ld0 * lfactor)
                lambda0 = np.identity(len(hes0), dtype=float) * ld0
                # lambda0 = np.diag(np.diag(hes0)) * ld0
                hinv0 = np.linalg.inv(hes0 + lambda0)
                n_iter += 1
                continue
            ld0 = max(1e-3, ld0 / lfactor)
            lambda0 = np.identity(len(hes0), dtype=float) * ld0

            # Adjust search boundary
            x0 = x1
            y0 = y1
            grad0 = gradient(func, x0)
            hes0 = hessian(func, x0)
            # lambda0 = np.diag(np.diag(hes0)) * ld0
            hinv0 = np.linalg.inv(hes0 + lambda0)
            n_iter += 1
        return None, n_iter

    # Find the minimum of a test function
    # Minimum should be [1.0, 1.0] - success
    y = lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    minx, iterations = lm(y, [1.0, 100.0])
    print(f"{minx=}")
    print(f"{iterations=}")

    # Find the minimum of a test function
    # Minimum should be [-0.70710677,  0.70710677] - success
    y = lambda x: 7 * float(x[0]) * x[1] / np.exp(x[0] ** 2 + x[1] ** 2)
    minx, iterations = lm(y, [-3.0, 3.0])
    print(f"{minx=}")
    print(f"{iterations=}")


def main() -> None:
    """Main program, used when run as a script."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("functions", nargs="*", help=f"Choose from {__all__}")
    args = parser.parse_args()

    functions = args.functions if args.functions else __all__
    for f in functions:
        if f not in __all__:
            raise ValueError(f'Invalid function "{f}" (choose from {__all__})')
        print("------", f'\nRunning "{f}"')
        globals()[f]()
        print("------")


main.__doc__ = __doc__


if __name__ == "__main__":
    main()
