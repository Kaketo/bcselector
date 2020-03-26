import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from bcselector.filter_methods.cost_based_filter_methods import difference_find_best_feature, fraction_find_best_feature
from bcselector.filter_methods.no_cost_based_filter_methods import no_cost_find_best_feature
from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife

__all__ = [
    '_MockVariableSelector',
    'DiffVariableSelector',
    'FractionVariableSelector'
]

class _MockVariableSelector():
    def __init__(self):
        self.data = None
        self.target_variable = None
        self.costs = None

        self.variables_selected_order = []
        self.cost_variables_selected_order = []
        self.j_criterion_func = None

        self.total_scores = None
        self.total_costs = None
        self.no_cost_total_scores = None
        self.no_cost_total_costs = None

        self.scoring = None

        self.fig = None
        self.ax = None

    def fit(self, data, target_variable, costs, j_criterion_func = 'cife', seed = 42, **kwargs):
        self.variables_selected_order = []
        self.cost_variables_selected_order = []

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

    def get_ranked_variables(self):
        return self.variables_selected_order
    
    def get_ranked_costs(self):
        return self.cost_variables_selected_order

    def scoreCV(self, model, scoring = 'roc_auc', cv = 4, seed=42, **kwargs):
        self.total_scores = []
        self.total_costs = []
        self.scoring = scoring
        
        assert len(self.variables_selected_order) > 0, "Run fit method first."
        current_cost = 0

        for i in range(1,len(self.variables_selected_order) + 1):
            cur_vars = self.variables_selected_order[0:i]
            score = cross_val_score(model, self.data[:,cur_vars], self.target_variable, scoring=scoring, cv=cv, **kwargs).mean()
            current_cost += self.costs[i-1]
            self.total_scores.append(score)
            self.total_costs.append(current_cost)

    def plot_scores(self, budget = None):
        assert len(self.total_scores) > 0, "Run scoreCV method first."
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        if budget is not None:
            assert isinstance(budget,(int,float)), "Argument `budget` must be float or int."
            self.ax.axvline(x=budget)
        self.ax.plot(self.total_costs, self.total_scores, linestyle='--', marker='o', color='b')
    
        self.ax.set_title('Model ' + self.scoring + ' vs cost' , fontsize = 14)
        self.ax.tick_params(size=14)
        self.ax.set_xlabel('Cost')
        self.ax.set_ylabel(self.scoring)

    def _no_cost_scoreCV(self, model, scoring = 'roc_auc', cv = 4, **kwargs):
        # Rank variables with NoCostVariableSelector
        S = set()
        U = set([i for i in range(self.data.shape[1])])

        variables_selected_order = []
        cost_variables_selected_order = []

        while len(U) > 0:
            k, _, cost = no_cost_find_best_feature(j_criterion_func = self.j_criterion_func, 
                                data = self.data, 
                                target_variable = self.target_variable, 
                                prev_variables_index = list(S),
                                possible_variables_index = list(U),
                                costs = self.costs)
            S.add(k)
            variables_selected_order.append(k)
            cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))

        current_cost = 0
        self.no_cost_total_scores = []
        self.no_cost_total_costs = []
        
        for i in range(1,len(variables_selected_order) + 1):
            cur_vars = variables_selected_order[0:i]
            score = cross_val_score(estimator = model, X = self.data[:,cur_vars], y = self.target_variable,**kwargs).mean()
            current_cost += self.costs[i-1]
            self.no_cost_total_scores.append(score)
            self.no_cost_total_costs.append(current_cost)

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
                                lamb = self.lamb,
                                **kwargs)
            S.add(k)
            self.variables_selected_order.append(k)
            self.cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))

    def plot_scores(self, budget = None, compare_no_cost_method = False, **kwargs):
        super().plot_scores(budget=budget)
        if compare_no_cost_method is True:
            assert 'model' in kwargs.keys(), "When comparing with no-cost method must select model for scoring."
            model = kwargs.pop('model')
            super()._no_cost_scoreCV(model = model, **kwargs)

            self.ax.plot(self.no_cost_total_costs, self.no_cost_total_scores, linestyle='--', marker='o', color='r', label = 'no regard to cost')
            self.ax.plot(self.total_costs, self.total_scores, linestyle='--', marker='o', color='b', label = 'with regard to costs')
            self.ax.legend()
        plt.show()

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
        # r
        assert isinstance(r, int) or isinstance(r, float), "Argument `r` must be integer or float"
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
                                r = self.r,
                                **kwargs)
            S.add(k)
            self.variables_selected_order.append(k)
            self.cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))
    
    def plot_scores(self, budget = None, compare_no_cost_method = False, **kwargs):
        super().plot_scores(budget=budget)
        if compare_no_cost_method is True:
            assert 'model' in kwargs.keys(), "When comparing with no-cost method must select model for scoring."
            model = kwargs.pop('model')
            super()._no_cost_scoreCV(model = model, **kwargs)

            self.ax.plot(self.no_cost_total_costs, self.no_cost_total_scores, linestyle='--', marker='o', color='r', label = 'no regard to cost')
            self.ax.plot(self.total_costs, self.total_scores, linestyle='--', marker='o', color='b', label = 'with regard to costs')
            self.ax.legend()
        plt.show()



class NoCostVariableSelector(_MockVariableSelector):
    """Ranks all features in dataset with difference cost filter method.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    def fit(self, data, target_variable, costs, j_criterion_func = 'cife', **kwargs):

        super().fit(data, target_variable, costs, j_criterion_func, **kwargs)
        
        S = set()
        U = set([i for i in range(self.data.shape[1])])

        self.variables_selected_order = []
        self.cost_variables_selected_order = []

        while len(U) > 0:
            k, _, cost = no_cost_find_best_feature(j_criterion_func = self.j_criterion_func, 
                                data = self.data, 
                                target_variable = self.target_variable, 
                                prev_variables_index = list(S),
                                possible_variables_index = list(U),
                                costs = self.costs)
            S.add(k)
            self.variables_selected_order.append(k)
            self.cost_variables_selected_order.append(cost)
            U = U.difference(set([k]))

    def plot_scores(self, model):
        super().plot_scores(model)
        plt.show()

        