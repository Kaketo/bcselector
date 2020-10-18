import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector
from bcselector.data_generation import MatrixGenerator, DataFrameGenerator

class TestMatrixGenerator(unittest.TestCase):
    def test_cife(self):
        # Given
        n_cols = 10
        n_rows = 100
        model = LogisticRegression()

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows, n_basic_cols=n_cols, noise_sigmas = [2,3], seed=2)
        lamb = 1
        beta = 0.5

        dvs = DiffVariableSelector()
        dvs.fit(data=X,
                target_variable=y,
                costs=costs,
                lamb=lamb,
                j_criterion_func='cife',
                budget = 5, 
                stop_budget=True,
                beta=beta)

        # Then
        self.assertGreater(len(costs), len(dvs.variables_selected_order))


if __name__ == '__main__':
    unittest.main()   