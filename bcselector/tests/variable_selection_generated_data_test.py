import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector
from bcselector.data_generation import MatrixGenerator, DataFrameGenerator

class TestMatrixGenerator(unittest.TestCase):
    def test_cife(self):
        # Given
        n_cols = 50
        n_rows = 100

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows, n_cols=n_cols, seed=0)
        lamb = 1
        beta = 0.5

        dvs = DiffVariableSelector()
        dvs.fit(data=X,
                target_variable=y,
                costs=costs,
                lamb=lamb,
                j_criterion_func='cife',
                beta=beta)

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))
        self.assertAlmostEqual(sum(costs), sum(dvs.cost_variables_selected_order))

   