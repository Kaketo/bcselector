import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector

class TestDiffVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')
        
        self.assertEqual(dvs.variables_selected_order[0], 2)

    def test_scoreCV(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        model = LinearRegression()
        dvs.scoreCV(model)
        
        self.assertEqual(len(dvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        dvs = DiffVariableSelector()
        model = LinearRegression()
        with self.assertRaises(AssertionError): dvs.scoreCV(model)

class TestFractionVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        self.assertIsInstance(fvs.variables_selected_order, list)
        self.assertEqual(len(fvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}
        r = 1
        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        self.assertIsInstance(fvs.variables_selected_order, list)
        self.assertEqual(len(fvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')
        
        self.assertEqual(fvs.variables_selected_order[0], 2)

    def test_scoreCV(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        model = LinearRegression()
        fvs.scoreCV(model)
        
        self.assertEqual(len(fvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        fvs = FractionVariableSelector()
        model = LinearRegression()
        with self.assertRaises(AssertionError): fvs.scoreCV(model)

class TestNoCostVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        self.assertIsInstance(ncvs.variables_selected_order, list)
        self.assertEqual(len(ncvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        self.assertIsInstance(ncvs.variables_selected_order, list)
        self.assertEqual(len(ncvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')
        
        self.assertEqual(ncvs.variables_selected_order[0], 2)

    def test_scoreCV(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        model = LinearRegression()
        ncvs.scoreCV(model)
        
        self.assertEqual(len(ncvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        ncvs = NoCostVariableSelector()
        model = LinearRegression()
        with self.assertRaises(AssertionError): ncvs.scoreCV(model)