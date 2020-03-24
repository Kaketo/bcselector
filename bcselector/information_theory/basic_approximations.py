import numpy as np

def entropy(vector, base=None):
    """
    This estimator computes the entropy of the empirical probability distribution.
    ----------
    vector : list or np.array
        Vector of which entropy is calculated. 
    base : int or float
        Base of the logarithm in entropy approximation
            - if None, np.e is selected and entropy is returned in nats.
    Returns
    -------
    vector_entropy : float
        Approximated entropy
    """
    assert isinstance(vector, (list)), "Argument 'vector' not in the right format"
    assert (isinstance(vector, np.ndarray) and len(vector.shape) == 1), "Argument 'vector' not in the right shape. Use numpy (n,) shape instead"

    if len(vector) == 0:
        "Entropy for one number is zero"
        return 0

    _,counts = np.unique(vector, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def conditional_entropy(vector,condition, base=None):
    """
    This estimator computes the conditional entropy of the empirical probability distribution.
    ----------
    vector : list or np.array
        Vector of which entropy is calculated. 
    condition: list or np.array
        Vector of condition for entropy.
    base : int or float
        Base of the logarithm in entropy approximation
            - if None, np.e is selected and entropy is returned in nats.
    Returns
    -------
    vector_entropy : float
        Approximated entropy.
    """
    assert isinstance(vector, (list)), "Argument 'vector' not in the right format."
    assert (isinstance(vector, np.ndarray) and len(vector.shape) == 1), "Argument 'vector' not in the right shape. Use numpy (n,) shape instead."
    assert isinstance(condition, (list)), "Argument 'condition' not in the right format."
    assert (isinstance(condition, np.ndarray) and len(vector.shape) == 1), "Argument 'condition' not in the right shape. Use numpy (n,) shape instead."

    if len(vector) == 0:
        "Entropy for one number is zero"
        return 0

    unique_condition_values = np.unique(condition)
    cond_entropy = 0
    for i in unique_condition_values:
        condition_proba = np.sum(condition == i) / len(condition)
        cond_entropy += entropy(vector[condition == i],base=base) * condition_proba
    return np.sum(cond_entropy)

def mutual_information(vector_1, vector_2, base=None):
    vector_1_entropy = entropy(vector=vector_1, base=base)
    cond_entropy = conditional_entropy(vector=vector_1, condition=vector_2, base=base)
    return vector_1_entropy - cond_entropy

def conditional_mutual_information(vector_1, vector_2, condition, base = None):
    unique_condition_values = np.unique(condition)
    cond_mutual_info = 0
    for i in unique_condition_values:
        condition_proba = np.sum(condition == i) / len(condition)
        cond_mutual_info += mutual_information(vector_1[condition == i], vector_2[condition == i], base=base) * condition_proba
    return np.sum(cond_mutual_info)