import unittest
import pandas as pd
import numpy as np
from bcselector.datasets import _discretize, load_sample, load_hepatitis


class TestDiscretization(unittest.TestCase):
    def test_basic(self):
        # Given
        x = np.random.normal(0, 1, 1000)
        # When
        x_discretized = _discretize(x, n_bins=20)
        # Then
        assert len(np.unique(x_discretized)) == 20


class TestSampleData(unittest.TestCase):
    def test_loading_data(self):
        # Given
        # When
        load_sample()

        # Then
    def test_data_shape(self):
        # Given
        # When
        X, y, costs = load_sample()

        # Then
        assert X.shape == (2000, 28)
        assert y.shape == (2000, )
        assert len(costs) == 28

    def test_data_type_frame(self):
        # Given
        # When
        X, y, costs = load_sample()

        # Then
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(costs, dict)

    def test_data_type_array(self):
        # Given
        # When
        X, y, costs = load_sample(as_frame=False)

        # Then
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(costs, list)


class TestHepatitis(unittest.TestCase):
    def test_loading_data(self):
        # Given
        # When
        load_hepatitis()

        # Then
    def test_data_shape(self):
        # Given
        # When
        X, y, costs = load_hepatitis()

        # Then
        assert X.shape == (155, 19)
        assert y.shape == (155, )
        assert len(costs) == 19

    def test_data_type_frame(self):
        # Given
        # When
        X, y, costs = load_hepatitis()

        # Then
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(costs, dict)

    def test_data_type_array(self):
        # Given
        # When
        X, y, costs = load_hepatitis(as_frame=False)

        # Then
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(costs, list)

    def test_data_discretization(self):
        # Given
        # When
        X1, y1, costs1 = load_hepatitis(as_frame=True, discretize_data=False)
        X2, y2, costs2 = load_hepatitis(as_frame=True, n_bins=10)

        # Then
        assert not np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert costs1 == costs2
        assert len(np.unique(X1.iloc[:, 0])) > len(np.unique(X2.iloc[:, 0]))
        assert [True if x != 1 else False for x in X1.nunique().values].count(True) / len(X1.nunique().values) == 1

    def test_data_colnames(self):
        # Given
        first_colname = 'age'
        middle_colname = 'spiders'
        last_colname = 'histology'
        # When
        X1, y1, costs1 = load_hepatitis(as_frame=True, discretize_data=False)
        X2, y2, costs2 = load_hepatitis(as_frame=True, n_bins=10)

        # Then
        assert X1.columns[0] == first_colname
        assert X2.columns[0] == first_colname
        assert X1.columns[10] == middle_colname
        assert X2.columns[10] == middle_colname
        assert X1.columns[-1] == last_colname
        assert X2.columns[-1] == last_colname
