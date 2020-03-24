import numpy as np
import warnings
from basic_approximations import entropy, conditional_entropy, mutual_information, conditional_mutual_information


def mim(data, target_variable, candidate_variable_index):
    """
    data - numpy matrix
    target_variable (Y) - numpy array with target variable
    candidate_variable_index (X_i) - index of candidate variable X_i in data matrix
    """

    candidate_variable = data[:,candidate_variable_index]
    return mutual_information(candidate_variable, target_variable)

def mifs(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    data - numpy matrix
    target_variable (Y) - numpy array with target variable
    prev_variables_index - index of previously selected variables in data matrix
    candidate_variable_index (X_i) - index of candidate variable X_i in data
    """

    if 'beta' not in kwargs.keys():
        beta = 1
        warnings.warn('Parameter \'beta\' not provided, default value of 1 is selected.')
    else:
        beta = kwargs.pop('beta')
        assert len(kwargs) == 0, 'Unused parameters ' + str(list(kwargs.keys()))

    candidate_variable = data[:,candidate_variable_index]
    
    redundancy_sum = 0
    for var in prev_variables_index:
        redundancy_sum += mutual_information(data[:,var], candidate_variable)
    
    return mutual_information(candidate_variable, target_variable) - beta*redundancy_sum

def mrmr(data, target_variable, prev_variables_index, candidate_variable_index):
    """
    data - numpy matrix
    target_variable (Y) - numpy array with target variable
    prev_variables_index - index of previously selected variables in data matrix
    candidate_variable_index (X_i) - index of candidate variable X_i in data
    """
    
    candidate_variable = data[:,candidate_variable_index]
    prev_variables_len = 1 if len(prev_variables_index) == 0 else len(prev_variables_index)
    
    redundancy_sum = 0
    for var in prev_variables_index:
        redundancy_sum += mutual_information(data[:,var], candidate_variable)
    
    return mutual_information(candidate_variable, target_variable) - 1/prev_variables_len*redundancy_sum


def jmi(data, target_variable, prev_variables_index, candidate_variable_index):
    """
    data - numpy matrix
    target_variable (Y) - numpy array with target variable
    prev_variables_index - index of previously selected variables in data matrix
    candidate_variable_index (X_i) - index of candidate variable X_i in data
    """

    candidate_variable = data[:,candidate_variable_index]
    prev_variables_len = 1 if len(prev_variables_index) == 0 else len(prev_variables_index)

    redundancy_sum = 0
    for var in prev_variables_index:
        
        a = mutual_information(data[:,var], candidate_variable)
        b = conditional_mutual_information(data[:,var], candidate_variable, target_variable)
        redundancy_sum += a - b

    return mutual_information(candidate_variable, target_variable) - 1/prev_variables_len*redundancy_sum

def cife(data, target_variable, prev_variables_index, candidate_variable_index, **kwargs):
    """
    data - numpy matrix
    target_variable (Y) - numpy array with target variable
    prev_variables_index - index of previously selected variables in data matrix
    candidate_variable_index (X_i) - index of candidate variable X_i in data
    """

    if 'beta' not in kwargs.keys():
        beta = 1
        warnings.warn('Parameter \'beta\' not provided, default value of 1 is selected.')
    else:
        beta = kwargs.pop('beta')
        assert len(kwargs) == 0, 'Unused parameters ' + str(list(kwargs.keys()))

    candidate_variable = data[:,candidate_variable_index]
    
    redundancy_sum = 0
    for var in prev_variables_index:
        a = mutual_information(data[:,var], candidate_variable)
        b = conditional_mutual_information(data[:,var], candidate_variable, target_variable)
        redundancy_sum += a - b
        
    return mutual_information(candidate_variable, target_variable) - beta*redundancy_sum