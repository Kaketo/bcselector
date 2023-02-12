import numpy as np
from pyitlib import discrete_random_variable as drv

__all__ = [
    "entropy",
    "conditional_entropy",
    "mutual_information",
    "conditional_mutual_information",
    "mutual_information_combined",
]


def entropy(vector, base=None):
    """This estimator computes the entropy of the empirical probability distribution.

    Parameters
    ----------
    vector: list or np.array
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

    assert isinstance(vector, (list)) or (
        isinstance(vector, np.ndarray) and len(vector.shape) == 1
    ), "Argument 'vector' not in the right shape. Use list or numpy (n,) shape instead"
    assert len(vector) > 0, "Argument 'vector' can't be empty"

    vector = np.array(vector)

    base = np.e if base is None else base

    if len(vector) == 1:
        "Entropy for one number is zero"
        return 0.0

    _, counts = np.unique(vector, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

    # return float(drv.entropy(vector, base=base))


def conditional_entropy(vector, condition, base=None):
    """This estimator computes the conditional entropy of the empirical probability distribution.

    Parameters
    ----------
    vector: list or np.array
        Vector of which entropy is calculated.
    condition: list or np.array
        Vector of condition for entropy.
    base: int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.

    Returns
    --------
    vector_entropy: float
        Approximated entropy.

    """
    assert isinstance(vector, (list)) or (
        isinstance(vector, np.ndarray) and len(vector.shape) == 1
    ), "Argument 'vector' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(condition, (list)) or (
        isinstance(condition, np.ndarray) and len(condition.shape) == 1
    ), "Argument 'condition' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(vector) > 0, "Argument 'vector' can't be empty"
    assert len(condition) > 0, "Argument 'condition' can't be empty"

    vector = np.array(vector)
    condition = np.array(condition)

    assert (
        vector.shape == condition.shape
    ), "Argument 'vector' must be the same lenght as 'condition'"

    base = np.e if base is None else base

    if len(vector) == 1:
        "Entropy for one number is zero"
        return 0.0

    # sort values to use np.split later
    vector_sorted = vector[condition.argsort()]
    condition_sorted = condition[condition.argsort()]

    binvalues = np.split(
        vector_sorted, np.unique(condition_sorted, return_index=True)[1][1:]
    )
    _, counts = np.unique(condition_sorted, return_counts=True)
    binprobas = counts / counts.sum()
    cond_entropy = 0

    for values, proba in zip(binvalues, binprobas):
        cond_entropy += entropy(values, base=base) * proba
    return cond_entropy

    # return float(drv.entropy_conditional(vector, condition, base=base))


def mutual_information(vector_1, vector_2, base=None):
    """
    This estimator computes the mutual information of two vectors with method of the empirical probability distribution.

    Parameters
    -----------
    vector_1 : list or np.array
        Vector of one variable.
    vector_2 : list or np.array
        Vector of one variable.
    base : int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.

    Returns
    --------
    variables_mutual_information: float
        Approximated mutual information between variables.

    """
    base = np.e if base is None else base
    vector_1_entropy = entropy(vector=vector_1, base=base)
    cond_entropy = conditional_entropy(vector=vector_1, condition=vector_2, base=base)
    return float(vector_1_entropy - cond_entropy)


def conditional_mutual_information(vector_1, vector_2, condition, base=None):
    """This estimator computes the conditional mutual information of two vectors and condition vector with method of
    the empirical probability distribution.

    Parameters
    -----------
    vector_1 : list or np.array
        Vector of one variable.
    vector_2: list or np.array
        Vector of one variable.
    condition: list or np.array
        Vector of condition for mutual information.
    base : int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.

    Returns
    --------
    variables_conditional_mutual_information : float
        Approximated conditional mutual information between variables.

    """
    assert isinstance(vector_1, list) or (
        isinstance(vector_1, np.ndarray) and len(vector_1.shape) == 1
    ), "Argument 'condition' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(vector_2, list) or (
        isinstance(vector_2, np.ndarray) and len(vector_2.shape) == 1
    ), "Argument 'condition' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(condition, list) or (
        isinstance(condition, np.ndarray) and len(condition.shape) == 1
    ), "Argument 'condition' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(vector_1) > 0, "Argument 'vector_1' can't be empty"
    assert len(vector_2) > 0, "Argument 'vector_2' can't be empty"
    assert len(condition) > 0, "Argument 'condition' can't be empty"

    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    condition = np.array(condition)

    assert (
        vector_1.shape == vector_2.shape == condition.shape
    ), "Argument 'vector_1' and 'vector_2' must be the same length as 'condition'"

    if len(condition) == 1:
        "Entropy for one number is zero"
        return 0.0

    base = np.e if base is None else base

    vector_1_sorted = vector_1[condition.argsort()]
    vector_2_sorted = vector_2[condition.argsort()]
    condition_sorted = condition[condition.argsort()]

    binvalues_1 = np.split(
        vector_1_sorted, np.unique(condition_sorted, return_index=True)[1][1:]
    )
    binvalues_2 = np.split(
        vector_2_sorted, np.unique(condition_sorted, return_index=True)[1][1:]
    )
    _, counts = np.unique(condition_sorted, return_counts=True)
    binprobas = counts / counts.sum()
    cond_mutual_info = 0

    for value_1, value_2, proba in zip(binvalues_1, binvalues_2, binprobas):
        cond_mutual_info += mutual_information(value_1, value_2, base=base) * proba
    return cond_mutual_info

    # return float(drv.information_mutual_conditional(vector_1, vector_2, condition, base))


def mutual_information_combined(X1, X2, Y, base=np.e):
    """This estimator computes the mutual information of two vectors combined (X1, X2) and another one vector Y
    with method of the empirical probability distribution.

    # I((X1, X2), Y) =
    # I(Y, X1) + I(Y, X1) + II(Y,X1,X2) =
    # I(Y, X1) + I(Y, X1) + I(Y,X1|X2) - I(Y,X1) =
    # I(Y, X1) + I(Y,X1|X2)

    Parameters
    -----------
    X1 : list or np.array
        Vector of one variable.
    X2: list or np.array
        Vector of one variable.
    Y: list or np.array
        Vector of y for mutual information.
    base : int or float
        Base of the logarithm in entropy approximation. If None, np.e is selected and entropy is returned in nats.
    Returns
    --------
    mutual_information_combined : float
        Approximated mutual information between variables.
    """
    assert isinstance(X1, list) or (
        isinstance(X1, np.ndarray) and len(X1.shape) == 1
    ), "Argument 'X1' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(X2, list) or (
        isinstance(X2, np.ndarray) and len(X2.shape) == 1
    ), "Argument 'Y' not in the right shape. Use list or numpy (n,) shape instead."
    assert isinstance(Y, list) or (
        isinstance(Y, np.ndarray) and len(Y.shape) == 1
    ), "Argument 'Z' not in the right shape. Use list or numpy (n,) shape instead."
    assert len(X1) > 0, "Argument 'X1' can't be empty"
    assert len(X2) > 0, "Argument 'X2' can't be empty"
    assert len(Y) > 0, "Argument 'Y' can't be empty"

    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)

    assert (
        X1.shape == X2.shape == Y.shape
    ), "Argument 'X1' and 'X2' must be the same length as 'Y'"

    a = mutual_information(vector_1=Y, vector_2=X2, base=base)
    b = conditional_mutual_information(vector_1=Y, vector_2=X1, condition=X2, base=base)

    # a = drv.information_mutual(Y, X2, base=base)
    # b = drv.information_mutual_conditional(X=Y, Y=X1, Z=X2, base=base)
    return a + b
