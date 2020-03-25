import numpy as np
import pandas as pd

from bcselector.filter_methods.cost_based_filter_methods import difference_find_best_feature, fraction_find_best_feature
from bcselector.filter_methods.no_cost_based_filter_methods import no_cost_find_best_feature
from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife

__all__ = [
    'DiffVariableSelector'
]

class _MockVariableSelector():
    def __init__(self):
        self.data = None
        self.target_variable = None
        self.costs = None

        self.variables_selected_order = None
        self.cost_variables_selected_order = None
        self.j_criterion_func = None

    def fit(self, data, target_variable, costs, j_criterion_func = 'cife', **kwargs):
        # data & costs
        assert isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame), "Argument `data` must be numpy.ndarray or pandas.DataFrame"
        if isinstance(data,np.ndarray):
            assert isinstance(costs,list), "When using `data` as np.array, provide `costs` as list of floats or integers"
        else:
            assert isinstance(costs,(list, dict)), "When using `data` as pd.DataFrame, provide `costs` as list of floats or integers or dict {'col_1':cost_1,...}"

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            if isinstance(costs,dict):
                self.costs = [costs[x] for x in data.columns]
            else:
                self.costs = costs
        else:
            self.data = data
            self.costs = costs

        assert len(self.data.shape) == 2, "For `data` argument use numpy array of shape (n,p) or pandas DataFrame" 
        assert data.shape[1] == len(costs), "Length od cost must equal number of columns in `data`"

        # target_variable
        assert isinstance(target_variable,np.ndarray) or isinstance(target_variable,pd.core.series.Series), "Use np.array or pd.Series for argument `target_variable`"

        if isinstance(target_variable,pd.core.series.Series):
            self.target_variable = target_variable.values
        else:
            self.target_variable = target_variable
        
        assert self.data.shape[0] == len(self.target_variable), "Number of rows in 'data' must equal target_variable length"

        # j_criterion_func
        j_criterion_dict = {'mim':mim,'mifs':mifs,'mrmr':mrmr,'jmi':jmi,'cife':cife}
        assert j_criterion_func in ['mim','mifs','mrmr','jmi','cife'], "Argument `j_criterion_func` must be one of ['mim','mifs','mrmr','jmi','cife']"
        self.j_criterion_func = j_criterion_dict[j_criterion_func]

class DiffVariableSelector(_MockVariableSelector):
    """Ranks all features in dataset with difference cost filter method.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    def fit(self, data, target_variable, costs, lamb, j_criterion_func = 'cife', **kwargs):
        # lamb
        assert isinstance(lamb, int) or isinstance(lamb, float), "Argument `lamb` must be integer or float"
        self.lamb = lamb
        super().fit(data, target_variable, costs, j_criterion_func, **kwargs)
        
        S = set()
        U = set([i for i in range(self.data.shape[1])])

        self.variables_selected_order = []
        self.cost_variables_selected_order = []

        while len(U) > 0:
            k, _, cost = difference_find_best_feature(j_criterion_func = self.j_criterion_func, 
                                data = self.data, 
                                target_variable = self.target_variable, 
                                prev_variables_index = list(S),
                                possible_variables_index = list(U),
                                costs = self.costs,
                                lamb = self.lamb)
            S.add(k)
            self.variables_selected_order.append(k)
            self.cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))

class FractionVariableSelector(_MockVariableSelector):
    """Ranks all features in dataset with difference cost filter method.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    def fit(self, data, target_variable, costs, r, j_criterion_func = 'cife', **kwargs):
        # lamb
        assert isinstance(r, int) or isinstance(r, float), "Argument `lamb` must be integer or float"
        self.r = r

        super().fit(data, target_variable, costs, j_criterion_func, **kwargs)
        
        S = set()
        U = set([i for i in range(self.data.shape[1])])

        self.variables_selected_order = []
        self.cost_variables_selected_order = []

        while len(U) > 0:
            k, _, cost = fraction_find_best_feature(j_criterion_func = self.j_criterion_func, 
                                data = self.data, 
                                target_variable = self.target_variable, 
                                prev_variables_index = list(S),
                                possible_variables_index = list(U),
                                costs = self.costs,
                                r = self.r)
            S.add(k)
            self.variables_selected_order.append(k)
            self.cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))

        