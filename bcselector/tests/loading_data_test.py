import unittest
import pandas as pd
import numpy as np
from bcselector.datasets import discretize, load_mimic3, load_hepatitis


class TestDiscretization(unittest.TestCase):
    def test_basic(self):
        # Given
        x = np.random.normal(0, 1, 1000)
        # When
        x_discretized = discretize(x, n_bins=20)
        # Then
        assert len(np.unique(x_discretized)) == 20


class TestMIMIC3(unittest.TestCase):
    def test_loading_data(self):
        # Given
        # When
        load_mimic3()

        # Then
    def test_data_shape(self):
        # Given
        # When
        X, y, costs = load_mimic3()

        # Then
        assert X.shape == (6591, 306)
        assert y.shape == (6591, 10)
        assert len(costs) == 306

    def test_data_type_frame(self):
        # Given
        # When
        X, y, costs = load_mimic3()

        # Then
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(costs, dict)

    def test_data_type_array(self):
        # Given
        # When
        X, y, costs = load_mimic3(as_frame=False)

        # Then
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(costs, list)

    def test_data_discretization(self):
        # Given
        # When
        X1, y1, costs1 = load_mimic3(as_frame=False, discretize_data=False)
        X2, y2, costs2 = load_mimic3(as_frame=False, n_bins=20)

        # Then
        assert not np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert costs1 == costs2
        assert len(np.unique(X1[:, 0])) > len(np.unique(X2[:, 0]))


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
        assert y.shape == (155, 1)
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
        X1, y1, costs1 = load_hepatitis(as_frame=False, discretize_data=False)
        X2, y2, costs2 = load_hepatitis(as_frame=False, n_bins=3)

        # Then
        assert not np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert costs1 == costs2
        assert len(np.unique(X1[:, 0])) > len(np.unique(X2[:, 0]))
