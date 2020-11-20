import numpy as np

def no_cost_find_best_feature(j_criterion_func, data, target_variable, possible_variables_index, costs, **kwargs):
    variables_result = []
    for i in possible_variables_index:
        variables_result.append(j_criterion_func(data, 
                                                target_variable = target_variable, 
                                                candidate_variable_index=i,
                                                **kwargs))
    k = np.argmax(variables_result)
    return possible_variables_index[k], variables_result[k], costs[possible_variables_index[k]]