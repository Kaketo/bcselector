import numpy as np

def fraction_find_best_feature(j_criterion_func, r, data, target_variable, possible_variables_index, costs, normalized_costs, **kwargs):
    """Function that ranks all features with selected j_criterion_func according to fraction method  and returns the feature with highest criterion value.

    Parameters
    ----------
    j_criterion_func : function
        Function from bcselector.information_theory.j_criterion_approximations
    r : float or int
        Scalling parameter (Impact of cost on whole approximation).
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints
        Indexes of previously selected variables.
    possible_variables_index : list of ints
        Index of all candidate variables in data matrix.
    costs : list of ints or floats
        List of costs of all variables in data matrix.
    **kwargs
        Other parameters passed to j_criterion_func
    Returns
    -------
    index_of_best_feature : int
        Index of best feature due to criterion.
    value_of_criterion : float
        Value of fraction_criterion for this feature.
    cost_of_best_feature : float or int
        Cost of best selected feature
    """
    criterion_values = []
    norm_costs_tmp = []
    for i in possible_variables_index:
        norm_cost = 0.000001 if normalized_costs[i] == 0 else normalized_costs[i]
        norm_costs_tmp.append(norm_cost)

        j_criterion_value = j_criterion_func(data, 
                                    target_variable=target_variable, 
                                    candidate_variable_index=i,
                                    **kwargs)
        criterion_values.append(j_criterion_value)

    # When any element of criterion_values is negative
    if any(i < 0 for i in criterion_values):
        m = abs(min(criterion_values))
        criterion_values = [i + m for i in criterion_values]
    else:
        m = 0
    
    filter_values = criterion_values.copy()
    for i, (var_score, norm_cost) in enumerate(zip(criterion_values, norm_costs_tmp)):
        filter_values[i] = var_score / norm_cost**r 
    k = np.argmax(filter_values)

    return possible_variables_index[k], filter_values[k], criterion_values[k] - m, costs[possible_variables_index[k]]

def difference_find_best_feature(j_criterion_func, lamb, data, target_variable, possible_variables_index, costs, normalized_costs, **kwargs):
    """Function that ranks all features with selected j_criterion_func according to difference method and returns the feature with highest criterion value.
    
    Parameters
    ----------
    j_criterion_func : function
        Function from bcselector.information_theory.j_criterion_approximations
    beta : float or int
        Scalling parameter (Impact of cost on whole approximation).
    data : np.array matrix
        Matrix of data set. Columns are variables, rows are observations.
    target_variable : int or float
        Target variable. Can not be in data!
    prev_variables_index: list of ints
        Indexes of previously selected variables.
    possible_variables_index : list of ints
        Index of all candidate variables in data matrix.
    costs : list of ints or floats
        List of costs of all variables in data matrix.
    **kwargs
        Other parameters passed to j_criterion_func
    Returns
    -------
    index_of_best_feature : int
        Index of best feature due to criterion.
    value_of_criterion : float
        Value of fraction_criterion for this feature.
    cost_of_best_feature : float or int
        Cost of best selected feature
    """
    criterion_values = []
    filter_values = []
    for i in possible_variables_index:
        j_criterion_value = j_criterion_func(data, 
                                    target_variable = target_variable, 
                                    candidate_variable_index=i,
                                    **kwargs)
        criterion_values.append(j_criterion_value)
        filter_values.append(j_criterion_value - lamb*normalized_costs[i])
    k = np.argmax(filter_values)
    return possible_variables_index[k], filter_values[k], criterion_values[k], costs[possible_variables_index[k]]


