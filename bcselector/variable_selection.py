from bcselector.filter_methods.cost_based_filter_methods import difference_find_best_feature, fraction_find_best_feature
from bcselector.filter_methods.no_cost_based_filter_methods import no_cost_find_best_feature

__all__ = [
    'DiffVariableSelector'
]

class DiffVariableSelector():
    """Ranks all features in dataset with difference cost filter method.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    def __init__(self, data, costs, target_variable):
        self.data = data
        self.target_variable = target_variable
        self.costs = costs