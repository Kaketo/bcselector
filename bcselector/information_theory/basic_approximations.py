import numpy as np
from pyitlib import discrete_random_variable as drv

__all__ = [
    'entropy',
    'entropy_conditional',
    'mutual_information',
    'mutual_information_conditional'
]


def entropy(x, base=np.e):
    """This estimator computes the entropy of the empirical probability distribution.

    Parameters
    ----------
    x: list or np.array
        Vector of which entropy is calculated.
    base: int or float (default=np.e)
        Base of the logarithm in entropy approximation

    Returns
    --------
    vector_entropy: float
        Approximated entropy

    Examples
    --------
    >>> from bcselector.information_theory.basic_approximations import entropy
    >>> foo = [1,4,1,2,5,6,3]
    >>> entropy(foo)

    """

    assert isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1), \
        "Argument 'x' not in the right shape. Use list or numpy (n,) shape instead "
    assert len(x) > 0, "Argument 'x' can't be empty"

    x = np.array(x)

    if len(x) == 1:
        "Entropy for one number is zero"
        return 0.0

    _, counts = np.unique(x, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    # return float(drv.entropy(x, base=base))


def entropy_conditional(x, y, base=np.e):
    """This estimator computes the conditional entropy of the empirical probability distribution.

    Parameters
    ----------
    x: list or np.array
        Vector of which entropy is calculated.
    y: list or np.array
        Vector of y for entropy.
    base: int or float (default=np.e)
        Base of the logarithm in entropy approximation.

    Returns
    --------
    vector_entropy: float
        Approximated entropy.

    """
    assert isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1), \
        "Argument 'x' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(y, list) or (isinstance(y, np.ndarray) and len(y.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(x) > 0, "Argument 'x' can't be empty"
    assert len(y) > 0, "Argument 'y' can't be empty"

    x = np.array(x)
    y = np.array(y)

    assert x.shape == y.shape, "Argument 'x' must be the same lenght as 'y'"

    if len(x) == 1:
        "Entropy for one number is zero"
        return 0.0

    # sort values to use np.split later
    vector_sorted = x[y.argsort()]
    condition_sorted = y[y.argsort()]

    binvalues = np.split(vector_sorted, np.unique(condition_sorted, return_index=True)[1][1:])
    _, counts = np.unique(condition_sorted, return_counts=True)
    binprobas = counts / counts.sum()
    cond_entropy = 0

    for values, proba in zip(binvalues, binprobas):
        cond_entropy += entropy(values, base=base) * proba
    return cond_entropy

    # return float(drv.entropy_conditional(x, y, base=base))


def mutual_information(x, y, base=np.e):
    """This estimator computes the mutual information of two vectors with method of the empirical probability distribution.

    Parameters
    -----------
    x : list or np.array
        Vector of one variable.
    y : list or np.array
        Vector of one variable.
    base : int or float (default=np.e)
        Base of the logarithm in entropy approximation.

    Returns
    --------
    variables_mutual_information: float
        Approximated mutual information between variables.

    """
    vector_1_entropy = entropy(x=x, base=base)
    cond_entropy = entropy_conditional(x=x, y=y, base=base)
    return float(vector_1_entropy - cond_entropy)


def mutual_information_conditional(x, y, z, base=np.e):
    """This estimator computes the conditional mutual information of two vectors and y x with method of the empirical probability distribution.

    Parameters
    -----------
    x : list or np.array
        Vector of one variable.
    y: list or np.array
        Vector of one variable.
    z: list or np.array
        Vector of y for mutual information.
    base : int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.

    Returns
    --------
    variables_conditional_mutual_information : float
        Approximated conditional mutual information between variables.

    """
    assert isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(y, list) or (isinstance(y, np.ndarray) and len(y.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(z, list) or (isinstance(z, np.ndarray) and len(z.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(x) > 0, "Argument 'x' can't be empty"
    assert len(y) > 0, "Argument 'y' can't be empty"
    assert len(z) > 0, "Argument 'y' can't be empty"

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert x.shape == y.shape == z.shape, "Argument 'x' and 'y' must be the same lenght as 'y'"

    if len(x) == 1:
        "Entropy for one number is zero"
        return 0.0

    vector_1_sorted = x[z.argsort()]
    vector_2_sorted = y[z.argsort()]
    condition_sorted = z[z.argsort()]

    binvalues_1 = np.split(vector_1_sorted, np.unique(condition_sorted, return_index=True)[1][1:])
    binvalues_2 = np.split(vector_2_sorted, np.unique(condition_sorted, return_index=True)[1][1:])
    _, counts = np.unique(condition_sorted, return_counts=True)
    binprobas = counts / counts.sum()
    cond_mutual_info = 0

    for value_1, value_2, proba in zip(binvalues_1, binvalues_2, binprobas):
        cond_mutual_info += mutual_information(value_1, value_2, base=base) * proba
    return cond_mutual_info

    # return float(drv.information_mutual_conditional(x, y, y, base))


def mutual_information_combined(x, y, z, base=np.e):
    """This estimator computes the conditional mutual information of two vectors and y x with method of the empirical probability distribution.

    Parameters
    -----------
    x : list or np.array
        Vector of one variable.
    y: list or np.array
        Vector of one variable.
    z: list or np.array
        Vector of y for mutual information.
    base : int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.

    Returns
    --------
    variables_conditional_mutual_information : float
        Approximated conditional mutual information between variables.

    """
    assert isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(y, list) or (isinstance(y, np.ndarray) and len(y.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(z, list) or (isinstance(z, np.ndarray) and len(z.shape) == 1), \
        "Argument 'y' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(x) > 0, "Argument 'x' can't be empty"
    assert len(y) > 0, "Argument 'y' can't be empty"
    assert len(z) > 0, "Argument 'y' can't be empty"

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert x.shape == y.shape == z.shape, "Argument 'x' and 'y' must be the same lenght as 'y'"

    a = mutual_information(x=z, y=y, base=base)
    b = mutual_information_conditional(x=z, y=x, z=y, base=base)

    # a = drv.information_mutual(z, y, base=base)
    # b = drv.information_mutual_conditional(X=z, Y=x, Z=y, base=base)
    return a + b
