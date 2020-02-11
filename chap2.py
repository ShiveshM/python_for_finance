#! /usr/bin/env python3

"""Utility functions from Chapter 2 of Mastering Python for Finance.

This module wraps into standalone functions the contents of Chapter 2 in James
Ma Weiming's "Mastering Python for Finance", published by Packt.
"""

__all__ = ['capm', 'apt', 'lin_programming', 'int_programming', 'linalg',
           'lu_decomp', 'cholesky', 'qr_decomp', 'jacobi', 'gauss_seidel']

STR_FMT = '{0}\n{1}\n'


def capm() -> None:
    """The Capital Asset Pricing Model (CAPM).

    The CAPM relates the risk premium of an asset and the market risk premium
    through the asset's beta:
                   E[R_i] - R_f = β(E[R_m] - R_f)
    where
        - R_f is the risk-free return e.g. interest rate on gov bonds
        - E[R_i] is the expected returns on the asset
        - E[R_m] is the expected return of the market
        - β is the sensitivity of the asset to the general market, extracted as
                    β = Cov[R_i, R_m] / Var[R_m]

    For a portfolio of assets, we can construct combinations or weights of
    risky securities that produce the lowest portfolio risk for every level of
    portfolio return. Combinations of optimal portfolios lie along a line
    called the *efficient frontier*. The tangent line which provides the best
    optimal portfolio is know as the *market portfolio*.

    There exists a straight line drawn from the market portfolio to the
    risk-free rate and is called the *Captial Market Line* (CML). This can be
    thought of as the highest Sharpe ratio available among all other Sharpe
    ratios of optimal portfolios.

    Another line of interest is the *Security Market Line* (SML), which plot
    the asset's expected returns against its beta. Securities priced above the
    SML are deemed to be undervalued, which securities below are deemed to be
    overvalued.

    CAPM suffers from limitations such as the fact that returns are captured by
    one risk factor - the market risk factor. In a well diversified portfolio,
    the unsystematic risk is essentially eliminated.

    """
    from scipy import stats

    # Data for 5 time periods
    stock_returns = [0.065, 0.0265, -0.0593, -0.001, 0.0346]
    market_returns = [0.055, -0.09, -0.041, 0.045, 0.022]

    # Calculate beta by regressing the stock returns to the market returns
    # R_i = α + β R_m
    help(stats.linregress)
    slope, intercept, rvalue, pvalue, stderr = \
        stats.linregress(stock_returns, market_returns)
    alpha = intercept
    beta  = slope
    print(STR_FMT.format('alpha',  alpha))
    print(STR_FMT.format('beta',   beta))
    print(STR_FMT.format('rvalue', rvalue))
    print(STR_FMT.format('pvalue', pvalue))
    print(STR_FMT.format('stderr', stderr))

    # For a risk-free rate of 5% and a market risk premium of 8.5%
    R_f  = 5
    R_am = 8.5 # R_am = E[R_m] - R_f
    E_R_i = R_f + (beta * R_am)
    print(STR_FMT.format('Expected return of asset', '{:.1f}%'.format(E_R_i)))


def apt() -> None:
    """The Arbitrage Pricing Theory model (APT)

    APT is a general theory of asset pricing that holds that the expected
    return of an asset can be modeled as a linear function of various
    factors, such as inflation rate, GDP growth rate, real interest rates, or
    dividends, where sensitivity to changes in each factor is represented by a
    factor-specific beta coefficient.
             R_i = α_i + β_i1 f_1 + β_i2 f_2 + ... + β_ij f_j + ϵ_i
          E[R_i] = R_f + β_i1 F_1 + β_i2 F_2 + ... + β_ij F_j
    where
        - R_i is the returns on asset i
        - α_i is a constant for asset i
        - β_ij is the sensitivity of the ith asset to factor j
        - f_i is a systematic factor
        - E[R_i] is the expected returns on asset i
        - R_f is the risk-free return e.g. interest rate on gov bonds
        - F_i is the risk premium of the factor

    Here we implement an APT model with seven factors using multilinear
    least squares regression techniques using the statsmodels package.

    """
    import numpy as np
    import statsmodels.api as sm

    # Generate some sample data
    num_periods = 9
    all_values = np.array([np.random.random(8)
                           for _ in range(num_periods)])

    # Filter the data
    y_values = all_values[:, 0]  # First column values as Y
    x_values = all_values[:, 1:] # Rest as X
    x_values = sm.add_constant(x_values) # Include the intercept
    help(sm.OLS)
    results = sm.OLS(y_values, x_values).fit() # Regress and fit the model
    print(STR_FMT.format('results.summary()', results.summary()))

    # Print intercept (R_f) and coefficients (betas)
    print(STR_FMT.format('results.params', results.params))


def lin_programming() -> None:
    """Use linear programming to determine portfolio allocation.

    Suppose we want to invest in two securities X and Y in such a way that we
    have three units of X for every two units of Y, such that the total number
    of units is maximised. However there are certain constraints:
        - For every two units of X and one unit of Y invested, the total volume
          must not exceed 100.
        - For every unit of the securities X and Y invested, the total volume
          must not exceed 80.
        - The total volume allowed to invest in the security X must not exceed
          40.
        - Short-selling is not allowed for securities.

    Therefore, mathematically we have:
        Maximise f(x, y) = 3x + 2y
    with
        - 2x + y <= 100
        - x + y <= 80
        - x <= 40
        - x, y >= 0

    """
    import pulp

    # Initialise pulp variables
    x = pulp.LpVariable('x', lowBound=0)
    y = pulp.LpVariable('y', lowBound=0)

    # Define the problem
    problem = pulp.LpProblem(
        'A_simple_maximisation_objective',
        pulp.LpMaximize
    )
    problem += 3*x + 2*y, 'The objective function'
    problem += 2*x + y <= 100, '1st constraint'
    problem += x + y <= 80, '2nd constraint'
    problem += x <= 40, '3rd constraint'

    # Solve the problem
    problem.solve()
    print('Maximisation results:')
    for variable in problem.variables():
        print(f'{variable.name} = {variable.varValue}')


def int_programming() -> None:
    """Use linear integer programming to determine portfolio allocation.

    Suppose we must go for 150 contracts in a particular OTC exotic security
    from three dealers.
    - Dealer X quoted $500 per contract plus handling fees of $4,000,
      regardless of the number of contracts sold.
    - Dealer Y charges $450 per contract plus a transaction fee of $2,000.
    - Dealer Z charges $450 per contract plus a fee of $6,000.
    - Dealer X will sell at most 100 contracts, dealer Y at most 90, and dealer
      Z at most 70.
    - The minimum transaction volume from any dealer is 30 contracts if any are
      transacted with that dealer.
    How should we minimise the cost of purchasing 150 contracts?

    """
    import pulp

    dealers = ['X', 'Y', 'Z']
    variable_costs = {'X': 500, 'Y': 350, 'Z': 450}
    fixed_costs = {'X': 4_000, 'Y': 2_000, 'Z': 6_000}

    # Initialise pulp variables
    # quantities defines the amount of contracts per dealer
    # is_orders defines whether or not a transaction is made per dealer
    quantities = pulp.LpVariable.dicts(
        'quantity', dealers, cat=pulp.LpInteger, lowBound=0, # Prevent shorting
    )
    is_orders = pulp.LpVariable.dicts(
        'orders', dealers, cat=pulp.LpBinary
    )

    # Define the problem
    # Minmise ∑_i {is_orders}_i ({variable_costs}_i × {quantities}_i
    #                            + {fixed_costs}_i)
    try:
        problem = pulp.LpProblem('Cost_minimisation_problem', pulp.LpMinimize)
        problem += sum([is_orders [i] * (variable_costs[i] * quantities[i] +
                                         fixed_costs[i])
                        for i in dealers]), 'Minimise portfolio cost'
        problem += sum([quantities[i] for i in dealers]) == 150, \
            'Total contracts required'
        problem += 30 <= quantities['X'] <= 100, 'Boundary of volume for X'
        problem += 30 <= quantities['Y'] <= 90, 'Boundary of volume for Y'
        problem += 30 <= quantities['Z'] <= 70, 'Boundary of volume for Z'
        problem.solve()
    except TypeError as exc:
        print(exc)
        print('As it turns out, we are trying to perform multiplication on '
        'two unknown variables, which led us to perform nonlinear '
        'programming!')

    # Define the problem
    # Another method is to reformulate the problem such that all unknown
    # variables are additive
    # Minmise ∑_i {variable_costs}_i × {quantities}_i
    #             + {fixed_costs}_i × {is_orders}_i
    # where the boundaries of quantities are now a function of is_orders
    problem = pulp.LpProblem('Cost_minimisation_problem', pulp.LpMinimize)
    problem += sum([variable_costs[i] * quantities[i]
                    + fixed_costs[i] * is_orders[i]
                    for i in dealers]), 'Minimise portfolio cost'
    problem += sum([quantities[i] for i in dealers]) == 150, \
        'Total contracts required'
    problem += is_orders['X'] * 30 <= quantities['X'] <= is_orders['X'] * 100,\
        'Boundary of volume for X'
    problem += is_orders['Y'] * 30 <= quantities['Y'] <= is_orders['Y'] * 90, \
        'Boundary of volume for Y'
    problem += is_orders['Z'] * 30 <= quantities['Z'] <= is_orders['Z'] * 70, \
        'Boundary of volume for Z'
    problem.solve()
    print('\nMinimisation results:')
    for variable in problem.variables():
        print(f'{variable.name} = {variable.varValue}')
    print(f'Total cost: {pulp.value(problem.objective)}')


def linalg() -> None:
    """Solving linear equations using matrices.

    If a set of systematic linear equations has constraints that are
    deterministic, we can represent the problem as matrices and apply matrix
    algebra.

    Suppose we want to build a portfolio that consists of three securities: a,
    b, and c. We have the following constraints:
    - It must consists of six units of a long position in security a.
    - With every two units of security a, one unit of security b, and one unit
      of security c must be invested, the net position must be four long units.
    - With every one unit of security a, three units of security b, and two
      units of security c invested, the net position must be five long units.

    This can be written in matrix form Ax = B with:
        [ 2 1 1 ]       [ a ]       [ 4 ]
    A = [ 1 3 2 ] , x = [ b ] , B = [ 5 ]
        [ 1 0 0 ]       [ c ]       [ 6 ]

    As the size of A increases, it becomes computationally expensive to compute
    the matrix inversion of A. One may consider other methods such as Cholesky
    decomposition, LU decomposition, QR decomposition, the Jacobi method, or
    the Gauss-Seidel method to break down A into simpler matrices for
    factorisation.

    """
    import numpy as np

    # Define A and B
    A = np.array([
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0]
    ])
    B = np.array([4, 5, 6])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Solve
    x = np.linalg.solve(A, B)
    print(STR_FMT.format('x',  x))


def lu_decomp() -> None:
    """The LU (lower-upper) decomposition.

    LU decomposition decomposes a matrix A into a lower triangular matrix, L,
    and an upper triangular matrix, U.
                         A = LU
    [ a b c ]   [ l_11 0    0   ]   [ u_11 u_12 u_13 ]
    [ d e f ] = [ l_21 l_22 0   ] × [ 0    u_22 u_23 ]
    [ g h i ]   [ l_31 l_32 l_33]   [ 0    0    u_33 ]

    This works for any square matrix (Cholesky decomposition works only for
    symmetric and positive definite matrices).

    """
    import numpy as np
    import scipy.linalg

    # Define A and B
    A = np.array([
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0]
    ])
    B = np.array([4, 5, 6])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Calculate pivoted LU decomposition and solve
    help(scipy.linalg.lu_factor)
    help(scipy.linalg.lu_solve)
    lu, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((lu, piv), B)
    print(STR_FMT.format('lu', lu))
    print(STR_FMT.format('piv', piv))
    print(STR_FMT.format('x', x))

    # Display LU decomposition of A where P is the permutation matrix, L is the
    # lower triangular matrix and U is the upper triangular matrix
    help(scipy.linalg.lu)
    P, L, U = scipy.linalg.lu(A)
    print(STR_FMT.format('P', P))
    print(STR_FMT.format('L', L))
    print(STR_FMT.format('U', U))


def cholesky() -> None:
    """The Cholesky decomposition.

    The matrix being decomposed must be Hermitian and positive definite.
                            A = L L^T*
    where L is the lower triangular matrix with real and positive numbers on
    the diagonals, and L^T* is the conjugate transpose of L.

    This method can be significantly faster and use less memory than LU
    decomposition, by exploting the property of symmetric matrices.

    """
    import numpy as np

    # Define A and B
    A = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ])
    B = np.array([6, 25, -11, 15])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Compute the Cholesky decomposition
    L = np.linalg.cholesky(A)
    print(STR_FMT.format('L', L))
    print(STR_FMT.format('np.dot(L, L.T.conj())', np.dot(L, L.T.conj())))

    # Solve for y = L^T* x, where L y = B
    y = np.linalg.solve(L, B)
    print(STR_FMT.format('y', y))

    # Finally solve for L^T* x = y
    x = np.linalg.solve(L.T.conj(), y)
    print(STR_FMT.format('x', x))
    print(STR_FMT.format('A @ x.T', A @ x.T))


def qr_decomp() -> None:
    """The QR decomposition.

    The matrix being decomposed is broken down into an orthogonal matrix, Q,
    and an upper triangular matrix, R:
                            A = Q R

    An orthogonal matrix Q:
    - Is square
    - Q Q^T = Q^T Q = I
    - Q^T = Q^{-1}

    To solve for Ax = B, Rx = Q^{-1} B = Q^T B. This is commonly used to solve
    least squares problems.

    """
    import numpy as np

    # Define A and B
    A = np.array([
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0]
    ])
    B = np.array([4, 5, 6])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Perform QR decomposition of A
    Q, R = np.linalg.qr(A)
    print(STR_FMT.format('Q', Q))
    print(STR_FMT.format('R', R))

    # Solve for y = Q^T B
    y = np.dot(Q.T, B)
    print(STR_FMT.format('y', y))

    # Solve for Rx = y
    x = np.linalg.solve(R, y)
    print(STR_FMT.format('x', x))


def jacobi() -> None:
    """The Jacobi method.

    This method solves a system of linear equations iteratively along its
    diagonal elements. The iteration terminates when the solution converges.

    The matrix is decomposed into two matrices of the same size such that
                            A = D + R
    where D consists of only the diagonal components of A, and the other matrix
    R consists of the remaining components. The solution is obtained
    iteratively,
    - Ax = B
    - (D + R)x = B
    - Dx = B - Rx
    - x_{n+1} = D^{-1}(B - Rx_n)

    If the A matrix is strictly irreducibly diagonally dominant, this method is
    guaranteed to converge. A strictly irreducibly diagonally dominant matrix
    is one where the absolute diagonal in every row is greater than the sum of
    the absolute values of other terms.

    """
    import numpy as np

    def jac(A, B, x=None, max_iter=int(1E6), tol=1E-10):
        """Solve for Ax = B using the Jacobi method."""
        # Create initial guess if needed
        if x is None:
            x = np.zeros_like(B)

        # Calculate D and R
        D = np.diag(A)
        R = A - np.diagflat(D)

        # Iterate until stopping condition
        for _ in range(max_iter):
            x_new = (B - np.dot(R, x)) / D
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        return x

    # Define A and B
    A = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ])
    B = np.array([6, 25, -11, 15])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Solve for x using Jacobi method
    x = jac(A, B)
    print(STR_FMT.format('x', x))


def gauss_seidel() -> None:
    """The Gauss-Seidel method.

    This method solves a system of linear equations iteratively. Here the
    matrix is decomposed into a lower triangular matrix, L, and an upper
    triangular matrix, U.
                            A = L + U
    The solution is obtained iteratively,
    - Ax = B
    - (L + U)x = B
    - Lx = B - Ux
    - x_{n+1} = L^{-1}(B - Ux_n)

    An advantage over the Jacobi method is that the elements of x_n can be
    overwritten in each iteration.

    """
    import numpy as np

    def g_s(A, B, x=None, max_iter=int(1E6), tol=1E-10):
        """Solve for Ax = B using the Gauss Seidel method."""
        # Create initial guess if needed
        if x is None:
            x = np.zeros_like(B)

        # Calculate L, U and L^{-1}
        L = np.tril(A) # returns the lower triangular matrix
        U = A - L
        L_inv = np.linalg.inv(L)

        # Iterate until stopping condition
        for _ in range(max_iter):
            Ux = np.dot(U, x)
            x_new = np.dot(L_inv, B - Ux)
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        return x

    # Define A and B
    A = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ])
    B = np.array([6, 25, -11, 15])
    print(STR_FMT.format('A', A))
    print(STR_FMT.format('B', B))

    # Solve for x using Gauss-Seidel method
    x = g_s(A, B)
    print(STR_FMT.format('x', x))


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Mastering Python for Finance - Chapter 2'
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
