import numpy as np
import warnings
from bcselector.information_theory.basic_approximations import mutual_information, conditional_mutual_information

__all__ = [
    'mim',
    'mifs',
    'mrmr',
    'jmi',
    'cife'
]


def mim(data, target_variable, candidate_variable_index, **kwargs):
    """
    This estimator computes the Mutual Information Maximisation criterion.

    Parameters
    ----------
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    candidate_variable_index : int
        Index of candidate variable in data matrix.

    Returns
    -------
    j_criterion_value : float
        J_criterion approximated by the Mutual Information Maximisation.
    """

    assert isinstance(data, np.ndarray), "Argument 'data' must be a numpy matrix"
    assert isinstance(target_variable, np.ndarray), "Argument 'target_variable' must be a numpy matrix"
    assert isinstance(candidate_variable_index, int), "Argument 'candidate_variable_index' must be an integer"

    assert len(data.shape) == 2, "For 'data' argument use numpy array of shape (n,p)"
    assert data.shape[0] == len(target_variable), "Number of rows in 'data' must equal target_variable length"
    assert candidate_variable_index < data.shape[1], "Index 'candidate_variable_index' out of range in 'data'"

    candidate_variable = data[:, candidate_variable_index]
    return mutual_information(candidate_variable, target_variable)


def mifs(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    This estimator computes the Mutual Information Feature Selection criterion.

    Parameters
    ----------
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints, set of ints
        Indexes of previously selected variables.
    candidate_variable_index : int
        Index of candidate variable in data matrix.
    beta: float
        Impact of redundancy segment in MIFS approximation. Higher the beta is, higher the impact.

    Returns
    -------
    j_criterion_value : float
        J_criterion approximated by the Mutual Information Feature Selection.
    """
    assert isinstance(data, np.ndarray), "Argument 'data' must be a numpy matrix"
    assert isinstance(target_variable, np.ndarray), "Argument 'target_variable' must be a numpy matrix"
    assert isinstance(candidate_variable_index, int), "Argument 'candidate_variable_index' must be an integer"

    assert len(data.shape) == 2, "For 'data' argument use numpy array of shape (n,p)"
    assert data.shape[0] == len(target_variable), "Number of rows in 'data' must equal target_variable length"
    assert candidate_variable_index < data.shape[1], "Index 'candidate_variable_index' out of range in 'data'"

    for i in prev_variables_index:
        assert isinstance(i, int), "All previous variable indexes must be int."

    if kwargs.get('beta') is None:
        beta = 1
        warnings.warn("Parameter `beta` not provided, default value of 1 is selected.", Warning)
    else:
        beta = kwargs.pop('beta')

    assert isinstance(beta, int) or isinstance(beta, float), "Argument 'beta' must be int or float"

    candidate_variable = data[:, candidate_variable_index]
    if len(prev_variables_index) == 0:
        redundancy_sum = 0
    else:
        redundancy_sum = np.apply_along_axis(mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable).sum()

    return mutual_information(candidate_variable, target_variable) - beta*redundancy_sum


def mrmr(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    This estimator computes the Max-Relevance Min-Redundancy criterion.

    Parameters
    ----------
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints
        Indexes of previously selected variables.
    candidate_variable_index : int
        Index of candidate variable in data matrix.

    Returns
    -------
    j_criterion_value : float
        J_criterion approximated by the Max-Relevance Min-Redundancy.
    """

    assert isinstance(data, np.ndarray), "Argument 'data' must be a numpy matrix"
    assert isinstance(target_variable, np.ndarray), "Argument 'target_variable' must be a numpy matrix"
    assert isinstance(candidate_variable_index, int), "Argument 'candidate_variable_index' must be an integer"

    assert len(data.shape) == 2, "For 'data' argument use numpy array of shape (n,p)"
    assert data.shape[0] == len(target_variable), "Number of rows in 'data' must equal target_variable length"
    assert candidate_variable_index < data.shape[1], "Index 'candidate_variable_index' out of range in 'data'"

    for i in prev_variables_index:
        assert isinstance(i, int), "All previous variable indexes must be int."

    candidate_variable = data[:, candidate_variable_index]
    prev_variables_len = 1 if len(prev_variables_index) == 0 else len(prev_variables_index)

    if len(prev_variables_index) == 0:
        redundancy_sum = 0
    else:
        redundancy_sum = np.apply_along_axis(mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable).sum()

    return mutual_information(candidate_variable, target_variable) - 1/prev_variables_len*redundancy_sum


def jmi(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    This estimator computes the Joint Mutual Information criterion.

    Parameters
    ----------
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints
        Indexes of previously selected variables.
    candidate_variable_index : int
        Index of candidate variable in data matrix.

    Returns
    -------
    j_criterion_value : float
        J_criterion approximated by the Joint Mutual Information.
    """

    assert isinstance(data, np.ndarray), "Argument 'data' must be a numpy matrix"
    assert isinstance(target_variable, np.ndarray), "Argument 'target_variable' must be a numpy matrix"
    assert isinstance(candidate_variable_index, int), "Argument 'candidate_variable_index' must be an integer"

    assert len(data.shape) == 2, "For 'data' argument use numpy array of shape (n,p)"
    assert data.shape[0] == len(target_variable), "Number of rows in 'data' must equal target_variable length"
    assert candidate_variable_index < data.shape[1], "Index 'candidate_variable_index' out of range in 'data'"

    for i in prev_variables_index:
        assert isinstance(i, int), "All previous variable indexes must be int."
    candidate_variable = data[:, candidate_variable_index]
    prev_variables_len = 1 if len(prev_variables_index) == 0 else len(prev_variables_index)

    if len(prev_variables_index) == 0:
        redundancy_sum = 0
    else:
        a = np.apply_along_axis(mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable).sum()
        b = np.apply_along_axis(conditional_mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable, condition=target_variable).sum()
        redundancy_sum = a - b

    return mutual_information(candidate_variable, target_variable) - 1/prev_variables_len*redundancy_sum


def cife(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    This estimator computes the Conditional Infomax Feature Extraction criterion.

    Parameters
    ----------
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints
        Indexes of previously selected variables.
    candidate_variable_index : int
        Index of candidate variable in data matrix.
    beta: float
        Impact of redundancy segment in MIFS approximation. Higher the beta is, higher the impact.

    Returns
    -------
    j_criterion_value : float
        J_criterion approximated by the Conditional Infomax Feature Extraction.
    """
    assert isinstance(data, np.ndarray), "Argument 'data' must be a numpy matrix"
    assert isinstance(target_variable, np.ndarray), "Argument 'target_variable' must be a numpy matrix"
    assert isinstance(candidate_variable_index, int), "Argument 'candidate_variable_index' must be an integer"

    assert len(data.shape) == 2, "For 'data' argument use numpy array of shape (n,p)"
    assert data.shape[0] == len(target_variable), "Number of rows in 'data' must equal target_variable length"
    assert candidate_variable_index < data.shape[1], "Index 'candidate_variable_index' out of range in 'data'"

    for i in prev_variables_index:
        assert isinstance(i, int), "All previous variable indexes must be int."

    if kwargs.get('beta') is None:
        beta = 1
        warnings.warn("Parameter `beta` not provided, default value of 1 is selected.")
    else:
        beta = kwargs.pop('beta')

    assert isinstance(beta, int) or isinstance(beta, float), "Argument 'beta' must be int or float"

    candidate_variable = data[:, candidate_variable_index]

    if len(prev_variables_index) == 0:
        redundancy_sum = 0
    else:
        a = np.apply_along_axis(mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable).sum()
        b = np.apply_along_axis(conditional_mutual_information, axis=0, arr=data[:, prev_variables_index], vector_2=candidate_variable, condition=target_variable).sum()
        redundancy_sum = a - b

    return mutual_information(candidate_variable, target_variable) - beta * redundancy_sum
