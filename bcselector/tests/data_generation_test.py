import unittest
import numpy as np
import pandas as pd

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector
from bcselector.data_generation import MatrixGenerator, DataFrameGenerator

class TestMatrixGenerator(unittest.TestCase):
    def test_basic_functions(self):
        # Given
        n_rows = 1000
        n_cols = 10
        seed = 0

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)

        # Then
        self.assertEqual(X.shape[0],n_rows)
        self.assertEqual(X.shape[1],n_cols)
        self.assertEqual(len(y), n_rows)
        self.assertEqual(len(costs), n_cols)

    def test_seed_repetition(self):
        # Given
        n_rows = 1000
        n_cols = 10
        seed = 0

        # When
        mg = MatrixGenerator()
        X_1, y_1, costs_1 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)
        X_2, y_2, costs_2 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)
        X_3, y_3, costs_3 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed + 1)

        # Then
        self.assertAlmostEqual(X_1.sum(), X_2.sum(), places = 5)
        self.assertNotAlmostEqual(X_1.sum(), X_3.sum(), places = 5)
        self.assertAlmostEqual(y_1.sum(), y_2.sum(), places = 5)
        self.assertNotAlmostEqual(y_1.sum(), y_3.sum(), places = 5)
        self.assertAlmostEqual(sum(costs_1), sum(costs_2), places = 5)
        self.assertNotAlmostEqual(sum(costs_1), sum(costs_3), places = 5)


class TestDataFrameGenerator(unittest.TestCase):
    def test_basic_functions(self):
        # Given
        n_rows = 1000
        n_cols = 10
        seed = 0

        # When
        mg = DataFrameGenerator()
        X, y, costs = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)

        # Then
        self.assertEqual(X.shape[0],n_rows)
        self.assertEqual(X.shape[1],n_cols)
        self.assertEqual(len(y), n_rows)
        self.assertEqual(len(costs), n_cols)

    def test_seed_repetition(self):
        # Given
        n_rows = 1000
        n_cols = 10
        seed = 0

        # When
        mg = DataFrameGenerator()
        X_1, y_1, costs_1 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)
        X_2, y_2, costs_2 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed)
        X_3, y_3, costs_3 = mg.generate(n_rows=n_rows,n_cols=n_cols,seed=seed + 1)
        costs_1 = sum(costs_1.values())
        costs_2 = sum(costs_2.values())
        costs_3 = sum(costs_3.values())


        # Then
        self.assertAlmostEqual(X_1.values.sum(), X_2.values.sum(), places = 5)
        self.assertNotAlmostEqual(X_1.values.sum(), X_3.values.sum(), places = 5)
        self.assertAlmostEqual(y_1.sum(), y_2.sum(), places = 5)
        self.assertNotAlmostEqual(y_1.sum(), y_3.sum(), places = 5)
        self.assertAlmostEqual(costs_1.sum(), costs_2.sum(), places = 5)
        self.assertNotAlmostEqual(costs_1.sum(), costs_3.sum(), places = 5)

if __name__ == '__main__':
    unittest.main()